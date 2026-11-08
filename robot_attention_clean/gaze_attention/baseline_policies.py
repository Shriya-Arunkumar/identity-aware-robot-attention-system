"""
baseline_policies.py — policies kept ONLY for comparison/ablation purposes,
never the active policy driving the robot by default.

`GreedyDwellPolicy` lives here (moved out of policies.py) specifically so
its baseline-only status is structural, not just a comment: it is not
imported by anything that builds the active/default pipeline
(main_webcam.py's `_make_active_policy` only ever calls it when a user
explicitly asks for `active_policy: greedy` / `--active-policy greedy` /
the `b` webcam hotkey — an explicit, deliberate choice, never a default).
`policies.py` re-exports the class (`from gaze_attention.baseline_policies
import GreedyDwellPolicy`) purely for backward-compatible imports and so it
can still appear in `POLICIES`/`make_policy` for baseline comparisons and
`evaluation/replay.py`'s filter x policy ablation grid — that registration
is exactly the "retained for comparison" use case, not activation as the
default.

Do not delete this code without updating every test and doc that references
it (see CLAUDE.md) — it is retained deliberately, not dead code.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from gaze_attention.dwell_arbitration import Observation, _DwellArbitration


class GreedyDwellPolicy(_DwellArbitration):
    """Baseline-only: ignores confidence beyond a simple gate. NEVER the
    active/default policy — see this module's docstring. Kept for
    comparison against ConversationalSimplePolicy (the active policy) and
    ConfidenceDebtPolicy in the evaluation/replay.py ablation grid.

    A person's score accumulates at a constant rate whenever they are
    seeking attention AND their confidence clears `confidence_threshold`
    (the gate) — the *magnitude* of confidence above that gate has no effect
    on how fast score accumulates, unlike ConfidenceDebtPolicy. Score decays
    at `engagement_decay` per second whenever the gate fails (not seeking, or
    confidence too low to trust).
    """

    BASELINE_ONLY = True  # structural marker: never construct this as the active policy

    SCORE_MAX = 20.0
    ACCUMULATION_RATE = 1.0  # score units per second while gated-seeking

    def __init__(
        self,
        confidence_threshold: float = 0.45,
        engagement_decay: float = 0.10,
        minimum_hold_seconds: float = 0.75,
        target_grace_seconds: float = 0.5,
        speech_mode: str = "off",
        speech_weight: float = 0.0,
        use_confidence: bool = True,
    ):
        super().__init__(
            minimum_hold_seconds=minimum_hold_seconds,
            target_grace_seconds=target_grace_seconds,
            speech_mode=speech_mode,
            speech_weight=speech_weight,
        )
        self.confidence_threshold = confidence_threshold
        self.engagement_decay = engagement_decay
        self._use_confidence = use_confidence
        self._score: Dict[int, float] = {}

    def _scores(self, observations: List[Observation], dt: float) -> Dict[int, float]:
        for obs in observations:
            pid = obs["id"]
            score = self._score.get(pid, 0.0)
            c_eff = obs["confidence"] if self._use_confidence else 1.0
            gated_seeking = obs["seeking_attention"] and c_eff >= self.confidence_threshold
            if gated_seeking:
                score += self.ACCUMULATION_RATE * dt
            else:
                score -= self.engagement_decay * dt
            self._score[pid] = float(max(0.0, min(self.SCORE_MAX, score)))
        return {obs["id"]: self._score[obs["id"]] for obs in observations}

    def _prune(self, present_ids: set, protect_id: Optional[int] = None) -> None:
        for pid in list(self._score):
            if pid not in present_ids and pid != protect_id:
                del self._score[pid]

    def reset(self) -> None:
        super().reset()
        self._score.clear()

    def get_metrics(self, person_id: int) -> Dict[str, float]:
        """Read-only per-person values for CSV logging/replay. Greedy has no
        separate engagement/debt concept, so `engagement` mirrors the
        pre-speech-bonus `score` and `debt` is always 0 — kept for a uniform
        CSV schema across policies. `score` includes the speech bonus (0.0
        when speech_mode == "off"), matching what selection actually used."""
        base_score = self._score.get(person_id, 0.0)
        bonus = self._last_speech_bonus.get(person_id, 0.0)
        return {"engagement": base_score, "debt": 0.0, "score": base_score + bonus, "speech_bonus": bonus}
