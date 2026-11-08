"""
filters.py — signal filters for the raw per-person (gaze_signal, confidence)
stream, swappable at runtime and via config (this week's ablation).

Common interface, identical across both filters:

    filtered_signal, filter_uncertainty = filter.update(person_id, gaze_signal, confidence, dt)

`gaze_signal` and `confidence` (the perception layer's RAW confidence — see
perception.py) are both already in [0, 1] by the time they reach a filter.

Scope note: a `KalmanFilter` existed earlier in this prototype and was
removed to keep the filter comparison focused on moving-average vs.
Bayesian filtering. This is a scope decision for this project's timeline,
not a claim that Kalman filtering is universally inferior — see README.md's
"Filter scope" section and DEBUG_ATTENTION.md (a historical diagnostic
snapshot from when Kalman was still active; kept as-is, not edited, since
it documents a real investigation at the time).

IMPORTANT — `filter_uncertainty` is NOT a confidence value and must never be
used as one (see DEBUG_ATTENTION.md, finding FM1, and the fix applied after
it). It answers "how uncertain is this filter about its own *smoothed
signal estimate* right now" — for the Bayesian filter that's literally the
posterior variance (normalized); for the moving average it's the sample
variance within the current window. It is a *filter-diagnostic* value,
useful for logging/debugging the filter itself, and rises and falls with
how erratic the recent signal has been — NOT with how trustworthy any
individual observation's raw confidence was. A filter that has been running
steadily for a while converges to LOW uncertainty even if every single
input observation had low raw confidence, because "steady" and
"trustworthy" are different things. `filter_uncertainty` is logged (see
logger.py) but is never fed into policy confidence gating — the value
policies actually gate on is `policy_confidence`, computed once directly
from raw perception confidence, upstream of any filter, in
main_webcam.py/evaluation.replay.py (see their docstrings for the current
"Version 1" transformation).

Filters must stay orthogonal to policies: nothing here knows about
engagement/debt/scores, and nothing in policies.py knows which filter ran.

Each filter keeps its own per-`person_id` state, created lazily on first
observation. Call `forget(person_id)` when a track is dropped (e.g. by
tracker.py's max_missed_frames) to avoid unbounded memory growth over a long
session; call `reset()` to clear all per-person state (used by the 'r'
keyboard control alongside a policy reset).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict

import numpy as np


class SignalFilter(ABC):
    """Interface every filter implements."""

    @abstractmethod
    def update(self, person_id: int, gaze_signal: float, confidence: float, dt: float) -> tuple[float, float]:
        """Advance person_id's filter state by one observation and return
        (filtered_signal, filter_uncertainty), both clamped to [0, 1].
        filter_uncertainty is a filter-diagnostic value, NOT a confidence —
        see this module's docstring."""

    @abstractmethod
    def forget(self, person_id: int) -> None:
        """Drop a person's filter state (their track ended)."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all per-person filter state."""


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


class MovingAverageFilter(SignalFilter):
    """Simple windowed smoothing — the baseline against which the
    model-based filter (Bayesian) is compared in the ablation.
    No confidence weighting at all: every sample in the window counts
    equally, which is exactly the failure mode the ablation is meant to
    expose (a confident and a near-blind frame smoothed identically).

    `filter_uncertainty` here is the sample variance of the signal within
    the current window — how much the raw signal has been bouncing around
    recently, not a reflection of the input confidence values (which this
    filter otherwise ignores entirely)."""

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._signal: Dict[int, Deque[float]] = {}

    def update(self, person_id: int, gaze_signal: float, confidence: float, dt: float) -> tuple[float, float]:
        sig_buf = self._signal.setdefault(person_id, deque(maxlen=self.window))
        sig_buf.append(_clip01(gaze_signal))
        filter_uncertainty = _clip01(float(np.var(sig_buf))) if len(sig_buf) > 1 else 0.0
        return float(np.mean(sig_buf)), filter_uncertainty

    def forget(self, person_id: int) -> None:
        self._signal.pop(person_id, None)

    def reset(self) -> None:
        self._signal.clear()


class BayesianFilter(SignalFilter):
    """Recursive Bayesian estimate of a single latent scalar (the person's
    "true" engagement level this moment), with confidence mapped to inverse
    observation noise.

    Each step:
      1. Predict: the belief mean relaxes toward `prior_mu` at `relax_rate`
         per second (mean reversion — "when in doubt, drift back to neutral"),
         and belief variance inflates by `process_noise * dt` (we grow less
         sure of a stale estimate over time).
      2. Correct: a standard 1D Bayesian update with observation noise
         R = R_min + (1 - confidence) * (R_max - R_min). Low confidence ->
         large R -> small gain -> the posterior stays close to the
         (already relaxed-toward-prior) prediction, i.e. the low-confidence
         observation is effectively ignored in favour of the prior. High
         confidence -> small R -> large gain -> the posterior follows the
         observation.

    `filter_uncertainty` is reported as the NORMALIZED POSTERIOR VARIANCE
    itself (higher = more uncertain about the current smoothed estimate),
    not a confidence and not its complement — see this module's docstring.
    A filter that has been running steadily converges to LOW uncertainty
    even if every input observation had low raw confidence; it must never
    be used as a substitute for the actual per-observation confidence.
    """

    def __init__(
        self,
        prior_mu: float = 0.3,
        prior_var: float = 0.25,
        relax_rate: float = 0.5,
        process_noise: float = 0.05,
        r_min: float = 0.01,
        r_max: float = 1.0,
        uncertainty_scale: float = 0.25,
    ):
        self.prior_mu = prior_mu
        self.prior_var = prior_var
        self.relax_rate = relax_rate
        self.process_noise = process_noise
        self.r_min = r_min
        self.r_max = r_max
        self.uncertainty_scale = uncertainty_scale
        self._state: Dict[int, tuple[float, float]] = {}  # id -> (mu, var)

    def update(self, person_id: int, gaze_signal: float, confidence: float, dt: float) -> tuple[float, float]:
        dt = max(dt, 1e-6)
        confidence = _clip01(confidence)
        mu, var = self._state.get(person_id, (self.prior_mu, self.prior_var))

        # --- Predict: relax toward prior, inflate uncertainty. ---------------
        mu_pred = mu + self.relax_rate * (self.prior_mu - mu) * dt
        var_pred = var + self.process_noise * dt

        # --- Correct: confidence -> observation noise -> Bayesian update. ----
        r = self.r_min + (1.0 - confidence) * (self.r_max - self.r_min)
        k = var_pred / (var_pred + r)
        mu_new = mu_pred + k * (gaze_signal - mu_pred)
        var_new = (1.0 - k) * var_pred

        self._state[person_id] = (mu_new, var_new)
        filter_uncertainty = _clip01(var_new / self.uncertainty_scale)
        return _clip01(mu_new), filter_uncertainty

    def forget(self, person_id: int) -> None:
        self._state.pop(person_id, None)

    def reset(self) -> None:
        self._state.clear()


FILTERS = {
    "moving_average": MovingAverageFilter,
    "bayesian": BayesianFilter,
}


def make_filter(name: str) -> SignalFilter:
    if name not in FILTERS:
        raise ValueError(f"Unknown filter {name!r}; choose from {list(FILTERS)}")
    return FILTERS[name]()
