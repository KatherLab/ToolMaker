from pathlib import Path

import pytest
from PIL import Image
from pytest_lazy_fixtures import lf
from toolmaker.run import ToolRunResult

from tests.utils import initialize, parametrize_invocation

initialize()


@parametrize_invocation("cucumber", "other_output_file", "png")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@pytest.mark.parametrize(
    "invocation,expected_segmentation_file",
    [
        (lf(test_case), filename)
        for (test_case, filename) in {
            "cucumber": "segmented_image.png",
            "other_output_file": "some_other_file.png",
            "png": "segmented_image.png",
        }.items()
    ],
)
def test_output_file(invocation: ToolRunResult, expected_segmentation_file: Path):
    segmentation_file = invocation.resolved_output_path / expected_segmentation_file
    assert segmentation_file.exists()
    assert segmentation_file.stat().st_size > 0
    img = Image.open(str(segmentation_file))
    assert img.size == (224, 224)  # all three images are 224x224
