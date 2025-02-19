import pytest
from pytest_lazy_fixtures import lf
from toolmaker.run import ToolRunResult

from tests.utils import initialize, parametrize_invocation

initialize()


@parametrize_invocation("crc_msi", "crc_braf", "crc_kras")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@parametrize_invocation("crc_msi", "crc_braf", "crc_kras")
def test_num_params(invocation: ToolRunResult):
    assert invocation.result["num_params"] == 3552258


@pytest.mark.parametrize(
    "invocation,expected_trained_model_path",
    [
        (lf("crc_msi"), "STAMP-CRC-MSI-model.pkl"),
        (lf("crc_braf"), "STAMP-CRC-BRAF-model.pkl"),
        (lf("crc_kras"), "STAMP-CRC-KRAS-model.pkl"),
    ],
)
def test_trained_model_exists(
    invocation: ToolRunResult, expected_trained_model_path: str
):
    trained_model_path = invocation.resolved_output_path / expected_trained_model_path
    assert trained_model_path.exists()
    assert trained_model_path.stat().st_size > 0
