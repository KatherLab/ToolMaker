from pathlib import Path

import numpy as np
import pytest
from pytest_lazy_fixtures import lf
from toolmaker.run import ToolRunResult

from tests.utils import TESTS_DATA_DIR, initialize, parametrize_invocation

initialize()


@parametrize_invocation("kather100k_muc", "tcga_brca_patch_png", "tcga_brca_patch_jpg")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@parametrize_invocation("kather100k_muc", "tcga_brca_patch_png", "tcga_brca_patch_jpg")
def test_shape_and_type(invocation: ToolRunResult):
    assert "features" in invocation.result
    features = invocation.result["features"]
    assert isinstance(features, list)
    assert len(features) == 1024
    assert all(isinstance(feature, float) for feature in features)


@pytest.mark.parametrize(
    "invocation,expected_features_file",
    [
        (lf(test_case), TESTS_DATA_DIR / filename)
        for (test_case, filename) in {
            "kather100k_muc": "uni_MUC-TCGA-ACCPKIPN.tif.npy",
            "tcga_brca_patch_png": "uni_TCGA-BRCA_patch_TCGA-BH-A0DE-01Z-00-DX1.64A0340A-8146-48E8-AAF7-4035988B9152.png.npy",
            "tcga_brca_patch_jpg": "uni_TCGA-BRCA_patch_TCGA-BH-A0DE-01Z-00-DX1.64A0340A-8146-48E8-AAF7-4035988B9152.jpg.npy",
        }.items()
    ],
)
def test_feature_values(invocation: ToolRunResult, expected_features_file: Path):
    np.testing.assert_allclose(
        np.array(invocation.result["features"], dtype=np.float32),
        np.load(expected_features_file).squeeze(),
        atol=1e-3,
    )
