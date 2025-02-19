from pathlib import Path

import numpy as np
import pytest
from pytest_lazy_fixtures import lf
from toolmaker.run import ToolRunResult

from tests.utils import TESTS_DATA_DIR, initialize, parametrize_invocation

initialize()


@parametrize_invocation("protein2", "protein2_with_mask", "protein3")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@parametrize_invocation("protein2", "protein2_with_mask", "protein3")
def test_type_sequence_representation(invocation: ToolRunResult):
    assert "sequence_representation" in invocation.result
    sequence_representation = invocation.result["sequence_representation"]
    assert isinstance(sequence_representation, list)
    assert all(isinstance(feature, float) for feature in sequence_representation)


@parametrize_invocation("protein2", "protein2_with_mask", "protein3")
def test_type_contact_map(invocation: ToolRunResult):
    assert "contact_map" in invocation.result
    contact_map = invocation.result["contact_map"]
    assert isinstance(contact_map, list)
    assert all(isinstance(row, list) for row in contact_map)
    assert all(isinstance(feature, float) for row in contact_map for feature in row)


@pytest.mark.parametrize(
    "invocation,expected_sequence_representation_file",
    [
        (lf(test_case), TESTS_DATA_DIR / filename)
        for (test_case, filename) in {
            "protein2": "esmfold_sequence_representations_protein2.npy",
            "protein2_with_mask": "esmfold_sequence_representations_protein2_with_mask.npy",
            "protein3": "esmfold_sequence_representations_protein3.npy",
        }.items()
    ],
)
def test_sequence_representation_values(
    invocation: ToolRunResult, expected_sequence_representation_file: Path
):
    np.testing.assert_allclose(
        np.array(invocation.result["sequence_representation"], dtype=np.float32),
        np.load(expected_sequence_representation_file).squeeze(),
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "invocation,expected_contact_map_file",
    [
        (lf(test_case), TESTS_DATA_DIR / filename)
        for (test_case, filename) in {
            "protein2": "esmfold_contacts_protein2.npy",
            "protein2_with_mask": "esmfold_contacts_protein2_with_mask.npy",
            "protein3": "esmfold_contacts_protein3.npy",
        }.items()
    ],
)
def test_contact_map_values(invocation: ToolRunResult, expected_contact_map_file: Path):
    np.testing.assert_allclose(
        np.array(invocation.result["contact_map"], dtype=np.float32),
        np.load(expected_contact_map_file).squeeze(),
        atol=1e-3,
    )
