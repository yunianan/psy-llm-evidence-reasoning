"""Restored project identity stays aligned with the existing patient simulator."""
import pytest
from pydantic import ValidationError

from layers.layer3.diagnosis.schemas import DiagnosticCandidate
from shared.disorder_catalog import canonical_disorder, disorder_metadata, treatment_key
from tests.tree_test_support import controlled_generation


@pytest.mark.parametrize("name", ["Insomnia", "insomnia", "Insomnia Disorder"])
def test_insomnia_names_resolve_to_restored_identity(name):
    assert canonical_disorder(name) == "Insomnia"
    metadata = disorder_metadata(name)
    assert metadata["disorder"] == "Insomnia"
    assert metadata["name"] == "Insomnia Disorder"
    assert metadata["label"] == "失眠障碍"
    assert metadata["knowledge_source"].endswith("#/Insomnia")


def test_insomnia_preserves_existing_treatment_route():
    assert treatment_key("Insomnia") == "Insomnia"


def test_current_candidate_contract_accepts_restored_id_and_rejects_d024():
    generation = controlled_generation(["Insomnia"], complete_labels=["Insomnia"])
    candidate = generation.candidates[0]
    assert candidate.disorder == "Insomnia"
    assert candidate.decision_paths[0].path_id == "Insomnia"
    payload = candidate.model_dump(mode="json")
    payload["disorder"] = "D024"
    with pytest.raises(ValidationError):
        DiagnosticCandidate.model_validate(payload)
