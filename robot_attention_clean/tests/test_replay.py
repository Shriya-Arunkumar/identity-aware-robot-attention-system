"""Synthetic replay gate: every scenario x every filter x every policy must
run start-to-finish with no NaN/Inf/crash. This must pass before the webcam
path is trusted (see project development order)."""

import pytest

from evaluation.replay import SCENARIOS, run_scenario
from gaze_attention import DEFAULT_CONFIG
from gaze_attention.filters import FILTERS
from gaze_attention.policies import POLICIES


@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
@pytest.mark.parametrize("filter_name", list(FILTERS))
@pytest.mark.parametrize("policy_name", list(POLICIES))
def test_combination_runs_clean(scenario_name, filter_name, policy_name):
    rows = run_scenario(SCENARIOS[scenario_name](), filter_name, policy_name, DEFAULT_CONFIG)
    assert len(rows) > 0
