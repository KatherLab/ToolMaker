from toolmaker.run import ToolRunResult

from tests.utils import initialize, parametrize_invocation

initialize()


@parametrize_invocation("kather100k_muc", "tcga_brca_patch_png", "tcga_brca_patch_jpg")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@parametrize_invocation("kather100k_muc", "tcga_brca_patch_png", "tcga_brca_patch_jpg")
def test_shape_and_type(invocation: ToolRunResult):
    assert "features" in invocation.result
    features = invocation.result["features"]
    assert isinstance(features, list)
    assert len(features) == 2048
    assert all(isinstance(feature, float) for feature in features)
