from pathlib import Path

import pytest
from PIL import Image
from pytest_lazy_fixtures import lf
from toolmaker.run import ToolRunResult

from tests.utils import initialize, parametrize_invocation

initialize()


@parametrize_invocation("prostate", "spleen")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@parametrize_invocation("prostate", "spleen")
def test_trained_model_exists(invocation: ToolRunResult):
    trained_model_path = invocation.resolved_output_path / "trained_model"
    assert trained_model_path.exists()
    assert trained_model_path.is_dir()
    assert any(f.is_file() for f in trained_model_path.glob("**/*"))
