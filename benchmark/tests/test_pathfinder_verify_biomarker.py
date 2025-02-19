import numpy as np
from toolmaker.run import ToolRunResult

from tests.utils import initialize, parametrize_invocation

initialize()


@parametrize_invocation("crc_tum_fraction_score", "crc_str_fraction_score")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@parametrize_invocation("crc_tum_fraction_score", "crc_str_fraction_score")
def test_types(invocation: ToolRunResult):
    assert "p_value" in invocation.result
    assert "hazard_ratio" in invocation.result
    assert isinstance(invocation.result["p_value"], float)
    assert isinstance(invocation.result["hazard_ratio"], float)


def test_pvalue_crc_tum(crc_tum_fraction_score: ToolRunResult):
    np.testing.assert_almost_equal(crc_tum_fraction_score.result["p_value"], 0.181042)


def test_pvalue_crc_str(crc_str_fraction_score: ToolRunResult):
    np.testing.assert_almost_equal(crc_str_fraction_score.result["p_value"], 0.005885)
