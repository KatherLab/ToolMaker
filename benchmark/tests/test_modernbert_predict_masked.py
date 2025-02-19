import pytest
from pytest_lazy_fixtures import lf
from toolmaker.run import ToolRunResult

from tests.utils import initialize, parametrize_invocation

initialize()


@parametrize_invocation("walking", "future_of_ai", "meaning_of_life")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@parametrize_invocation("walking", "future_of_ai", "meaning_of_life")
def test_prediction(invocation: ToolRunResult):
    assert "prediction" in invocation.result
    prediction = invocation.result["prediction"]
    assert isinstance(prediction, str)


@pytest.mark.parametrize(
    "invocation,original_part",
    [
        (lf("walking"), "he walked to the"),
        (lf("future_of_ai"), "the future of ai is"),
        (lf("meaning_of_life"), "the meaning of life is"),
    ],
)
def test_prediction_contains_original_sentence(
    invocation: ToolRunResult, original_part: str
):
    assert invocation.result["prediction"].lower().startswith(original_part)
