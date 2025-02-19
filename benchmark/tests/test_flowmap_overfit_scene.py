import pytest
from pytest_lazy_fixtures import lf
from toolmaker.run import ToolRunResult

from tests.utils import initialize, parametrize_invocation

initialize()


@parametrize_invocation("llff_fern", "llff_orchids")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@pytest.mark.parametrize(
    "invocation",
    [
        lf("llff_fern"),
        lf("llff_orchids"),
    ],
)
def test_types_and_shapes(invocation: ToolRunResult):
    assert "camera_extrinsics" in invocation.result
    assert "n" in invocation.result

    camera_extrinsics = invocation.result["camera_extrinsics"]
    n = invocation.result["n"]

    assert isinstance(n, int)
    assert isinstance(camera_extrinsics, list)
    for extrinsic in camera_extrinsics:
        assert isinstance(extrinsic, list)
        assert len(extrinsic) == 4
        for row in extrinsic:
            assert isinstance(row, list)
            assert len(row) == 4
            for element in row:
                assert isinstance(element, float)


@pytest.mark.parametrize(
    "invocation,expected_n",
    [
        (lf("llff_fern"), 20),
        (lf("llff_orchids"), 25),
    ],
)
def test_correct_number_of_frames(invocation: ToolRunResult, expected_n: int):
    assert "n" in invocation.result
    assert "camera_extrinsics" in invocation.result
    n = invocation.result["n"]
    camera_extrinsics = invocation.result["camera_extrinsics"]
    assert n == expected_n
    assert len(camera_extrinsics) == expected_n
