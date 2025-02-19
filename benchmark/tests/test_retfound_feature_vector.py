from toolmaker.run import ToolRunResult

from tests.utils import initialize, parametrize_invocation

initialize()


@parametrize_invocation("jpg", "png", "cucumber_different_filename")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@parametrize_invocation("jpg", "png", "cucumber_different_filename")
def test_shape_and_type(invocation: ToolRunResult):
    assert isinstance(invocation.result["feature_vector"], list)
    assert len(invocation.result["feature_vector"]) == 1024
    assert all(
        isinstance(feature, float) for feature in invocation.result["feature_vector"]
    )
