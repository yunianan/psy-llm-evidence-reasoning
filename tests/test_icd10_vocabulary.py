from icd10_tree import build_catalog

from layers.layer2.profile.ingest import ProfileIngestor
from layers.layer2.schemas import L2SessionState, SemanticAnalysis
from layers.layer2.symptom_vocabulary import ICD10_SYMPTOM_TERMS


def test_l2_accepts_icd10_symptoms_and_preserves_episode_context():
    state = L2SessionState(session_id="catalog-test", safety_passport={}, turn_index=1)
    analysis = SemanticAnalysis.model_validate({
        "input_type": "free_narrative",
        "symptoms": [
            {"value": "careless_errors", "evidence_quote": "我经常漏看题目"},
            {"value": "S001", "evidence_quote": "我经常漏看题目"},
        ],
        "structured_fields": [{"field": "diagnostic_details", "value": {"onset_age": 10},
                               "evidence_quote": "十岁时就这样"}],
    })
    ProfileIngestor().ingest(state, analysis=analysis, patient_text="我经常漏看题目，十岁时就这样",
                            message_id="m1")
    assert set(state.symptoms) == {"careless_errors"}
    assert state.symptoms["careless_errors"].patient_quotes == ["我经常漏看题目"]
    assert state.diagnostic_context.diagnostic_details == {"onset_age": 10}
    assert state.diagnostic_context.sources["diagnostic_details"]


def test_l2_vocabulary_sources_are_in_the_local_icd10_catalog_or_cddg():
    catalog = build_catalog()
    assert catalog["standard"] == "WHO ICD-10"
    for label, source in ICD10_SYMPTOM_TERMS.values():
        assert label
        if source == "WHO_CDDG_1992 F30.1 PDF p.127":
            assert catalog["documents"]["WHO_CDDG_1992"]["isbn"] == "9241544228"
        else:
            node = catalog["nodes"][source]
            assert node["original_clause"] and "calculation" not in node
