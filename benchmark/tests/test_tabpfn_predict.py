import pytest
from pytest_lazy_fixtures import lf
from toolmaker.run import ToolRunResult

from tests.utils import initialize, parametrize_invocation

initialize()


@parametrize_invocation("diabetes", "heart_disease", "parkinsons")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@parametrize_invocation("diabetes", "heart_disease", "parkinsons")
def test_types(invocation: ToolRunResult):
    assert "roc_auc" in invocation.result
    assert "accuracy" in invocation.result
    assert "probs" in invocation.result
    assert isinstance(invocation.result["roc_auc"], float)
    assert isinstance(invocation.result["accuracy"], float)
    assert isinstance(invocation.result["probs"], list)
    assert all(isinstance(p, float) for p in invocation.result["probs"]) or all(
        isinstance(prob, list) and all(isinstance(p, float) for p in prob)
        for prob in invocation.result["probs"]
    )  # either (N,) or (N, 2)


@pytest.mark.parametrize(
    "invocation,expected_n",
    [
        (lf("diabetes"), 154),
        (lf("heart_disease"), 54),
        (lf("parkinsons"), 39),
    ],
)
def test_number_of_probs(invocation: ToolRunResult, expected_n: int):
    assert len(invocation.result["probs"]) == expected_n
    assert all(isinstance(prob, float) for prob in invocation.result["probs"]) or all(
        len(prob) == 2 for prob in invocation.result["probs"]
    )
