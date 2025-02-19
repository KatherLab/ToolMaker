import h5py
import numpy as np
import pytest
from pytest_lazy_fixtures import lf
from toolmaker.run import ToolRunResult

from tests.utils import initialize, parametrize_invocation

initialize()


@parametrize_invocation("crc", "crc_single", "brca_single")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@pytest.mark.parametrize(
    "invocation,expected_num_processed_slides",
    [
        (lf("crc"), 10),
        (lf("crc_single"), 1),
        (lf("brca_single"), 1),
    ],
)
def test_num_processed_slides(
    invocation: ToolRunResult, expected_num_processed_slides: int
):
    assert invocation.result["num_processed_slides"] == expected_num_processed_slides


@pytest.mark.parametrize(
    "invocation,expected_num_files",
    [
        (lf("crc"), 10),
        (lf("crc_single"), 1),
        (lf("brca_single"), 1),
    ],
)
def test_output_files_exist(invocation: ToolRunResult, expected_num_files: int):
    assert invocation.resolved_output_path.exists()
    assert (
        len(list(invocation.resolved_output_path.glob("**/*.h5"))) == expected_num_files
    )


@parametrize_invocation("crc", "crc_single", "brca_single")
def test_output_files_have_correct_shape_and_type(invocation: ToolRunResult):
    for h5_file in invocation.resolved_output_path.glob("*.h5"):
        with h5py.File(h5_file, "r") as f:
            assert "features" in f
            features = f["features"]
            assert features.shape == (512,)
            assert features.dtype == np.float32
