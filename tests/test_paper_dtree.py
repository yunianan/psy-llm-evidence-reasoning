"""Behavioral tests for the independent paper workflow; all diagnoses are fictitious."""
import builtins
import copy
import importlib
import json
from pathlib import Path

import pytest

from paper_dtree.engine import DecisionTreeRunner, PAIRWISE_POLICY, RetryPolicy, RunFailed
from paper_dtree.models import Bundle, Response, load_bundle


SOURCE = {"document": "engineering-fixture", "print_pages": [1], "pdf_pages": [1],
          "note": "Nonclinical engineering fixture; not handbook text"}


def node(question, *, yes=None, no=None):
    return {"question": question, "yes": yes or {}, "no": no or {}, "source": copy.deepcopy(SOURCE)}


def tree(key, nodes, roots=None):
    return {"id": key, "title": f"Fixture {key}", "screening_question": f"screen-{key}",
            "screening_source": copy.deepcopy(SOURCE), "roots": roots or [next(iter(nodes))], "nodes": nodes}


def bundle_data(*trees, stage="refined", diagnoses=None):
    return {"id": "engineering-only", "standard": "DSM-5", "stage": stage, "coverage": "partial",
            "documents": ["engineering-fixture"], "diagnoses": diagnoses or ["Dx-A", "Dx-B", "Dx-C", "Dx-Other"],
            "review_status": "source_verified", "trees": list(trees),
            "refinements": [{"tree_id": trees[0]["id"], "node_id": "structure", "category": "fixture",
                              "before": "single candidate", "after": "multiple candidates", "reason": "test only",
                              "source": copy.deepcopy(SOURCE)}] if stage == "refined" else []}


class Scripted:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return Response(answer, 10, 2, "fixture-model", str(len(self.requests)))


def runner(data, transport, **kwargs):
    return DecisionTreeRunner(Bundle.parse(data), transport, model="fixture-model",
                              pairwise_policy=PAIRWISE_POLICY,
                              retry=kwargs.pop("retry", RetryPolicy(max_attempts=2, initial_seconds=0)),
                              **kwargs)


def test_all_entries_screened_then_binary_branches_then_every_candidate_pair():
    data = bundle_data(
        tree("T1", {"a": node("a", yes={"diagnoses": ["Dx-A"], "next": ["b"]}),
                    "b": node("b", yes={"diagnoses": ["Dx-B"]})}),
        tree("T2", {"c": node("c", yes={"diagnoses": ["Dx-C"]})}))
    api = Scripted("yes", "no", "yes", "yes", '["Dx-B"]')
    result = runner(data, api).run("raw case", case_id="case-1")
    assert [r.messages[1]["content"].rsplit("The question is as follows: ", 1)[-1]
            for r in api.requests[:4]] == ["screen-T1.", "screen-T2.", "a.", "b."]
    assert result["candidates_before_pairwise"] == ["Dx-A", "Dx-B"]
    assert result["diagnoses_after_pairwise"] == ["Dx-B"]
    assert result["postprocessing"] == "pending_simplification_and_reconciliation"
    for req in api.requests:
        assert len(req.messages) == 2 and [m["role"] for m in req.messages] == ["system", "user"]
        assert "raw case" in req.messages[1]["content"]
        assert (req.temperature, req.top_p, req.frequency_penalty, req.presence_penalty) == (0, 1, 0, 0)
        assert "DSM-5 diagnoses" in req.messages[0]["content"]
        assert "confidence" not in req.messages[1]["content"]


def test_all_selected_trees_advance_from_shared_frontier_then_compare_all_candidates():
    data = bundle_data(
        tree("T1", {"a": node("a", yes={"next": ["b"], "diagnoses": ["Dx-A"]}),
                    "b": node("b", yes={"diagnoses": ["Dx-B"]})}),
        tree("T2", {"c": node("c", yes={"next": ["d"], "diagnoses": ["Dx-C"]}), "d": node("d")}))
    api = Scripted("yes", "yes", "yes", "yes", "yes", "yes",
                   '["Dx-A","Dx-C"]', '["Dx-A","Dx-B"]', '["Dx-C","Dx-B"]')
    events = []
    result = runner(data, api, record=events.append).run("case", case_id="c")
    assert [(r["tree_id"], r["node_id"]) for r in result["tree_trace"]] == [
        ("T1", "a"), ("T2", "c"), ("T1", "b"), ("T2", "d")]
    assert [e["stage"] for e in events if e["event"] == "request_attempt"] == [
        "screening", "screening", "node", "node", "node", "node", "pairwise", "pairwise", "pairwise"]
    assert result["candidates_before_pairwise"] == result["raw_diagnoses"] == ["Dx-A", "Dx-C", "Dx-B"]
    assert result["traversal_policy"] == "screen_all_then_shared_node_frontier_v3"


@pytest.mark.parametrize("raw,expected", [
    ('["dx-a"]', ["Dx-A"]),
    ('[" DX-A ","dx-b"]', ["Dx-A", "Dx-B"]),
])
def test_pairwise_accepts_case_and_outer_space_variants_but_retains_original_response(raw, expected):
    data = bundle_data(tree("T", {"q": node("q", yes={"diagnoses": ["Dx-A", "Dx-B"]})}))
    events = []
    api = Scripted("yes", "yes", raw)
    result = runner(data, api, record=events.append).run("case", case_id="c")
    assert result["raw_diagnoses"] == expected and len(api.requests) == 3
    event = next(e for e in events if e.get("stage") == "pairwise")
    assert event["response"]["text"] == raw and event["parsed"] == expected


@pytest.mark.parametrize("invalid", ['["dx-a","DX-A"]', '["Dx-A disorder"]'])
def test_pairwise_case_binding_rejects_duplicates_and_semantic_name_changes(invalid):
    data = bundle_data(tree("T", {"q": node("q", yes={"diagnoses": ["Dx-A", "Dx-B"]})}))
    api = Scripted("yes", "yes", invalid, '["Dx-A"]')
    result = runner(data, api).run("case", case_id="c")
    assert result["raw_diagnoses"] == ["Dx-A"] and len(api.requests) == 4
    assert api.requests[-1] == api.requests[-2]


def test_yes_and_no_choose_distinct_branches_and_unknown_is_instructed_as_no():
    data = bundle_data(tree("T", {"q": node("question", yes={"diagnoses": ["Dx-A"]},
                                          no={"diagnoses": ["Dx-B"]})}))
    api = Scripted("yes", "no")
    result = runner(data, api).run("case with missing facts", case_id="c")
    assert result["raw_diagnoses"] == ["Dx-B"]
    assert "If there is insufficient information, answer 'no'." in api.requests[1].messages[1]["content"]


def test_invalid_binary_is_retried_independently_without_turning_into_no():
    data = bundle_data(tree("T", {"q": node("q", yes={"diagnoses": ["Dx-A"]})}))
    api, events, sleeps = Scripted("pending", "YES", "yes"), [], []
    result = runner(data, api, record=events.append, sleep=sleeps.append).run("case", case_id="c")
    assert result["raw_diagnoses"] == ["Dx-A"]
    assert api.requests[0] == api.requests[1]
    attempts = [e for e in events if e["event"] == "request_attempt"]
    assert [e["status"] for e in attempts] == ["failed", "success", "success"]
    assert attempts[0]["response"]["text"] == "pending"
    assert len(sleeps) == 1


def test_exhausted_transport_error_is_failure_not_a_negative_screen_or_empty_prediction():
    data = bundle_data(tree("T", {"q": node("q")}))
    api, events = Scripted(ConnectionError("transport down"), ConnectionError("still down")), []
    with pytest.raises(RunFailed):
        runner(data, api, record=events.append).run("case", case_id="c")
    finished = events[-1]["result"]
    assert finished["status"] == "failed" and finished["screening"] == []
    assert "raw_diagnoses" not in finished


def test_default_retry_until_valid_has_exponential_backoff_and_counts_no_extra_samples():
    data = bundle_data(tree("T", {"q": node("q")}))
    api, sleeps = Scripted("invalid", "invalid", "invalid", "no"), []
    result = runner(data, api, retry=RetryPolicy(), sleep=sleeps.append).run("case", case_id="c")
    assert sleeps == [1, 2, 4]
    assert len(result["screening"]) == 1 and result["raw_diagnoses"] == []
    assert result["status"] == "success"


def test_interrupt_is_recorded_and_is_not_retried():
    data = bundle_data(tree("T", {"q": node("q")}))
    events = []

    class Interrupted:
        def complete(self, request):
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        runner(data, Interrupted(), record=events.append).run("case", case_id="c")
    assert events[-1]["result"]["status"] == "interrupted"


def test_multiple_roots_and_shared_node_keep_multiple_diagnoses_without_duplicate_question():
    data = bundle_data(tree("T", {
        "a": node("a", yes={"diagnoses": ["Dx-A"], "next": ["c"]}),
        "b": node("b", yes={"diagnoses": ["Dx-A"], "next": ["c"]}),
        "c": node("c", yes={"diagnoses": ["Dx-B"]}),
    }, roots=["a", "b"]))
    result = runner(data, Scripted("yes", "yes", "yes", "yes", '["Dx-A","Dx-B"]')).run("case", case_id="c")
    assert len(result["tree_trace"]) == 3
    assert len(result["candidate_sources"]["Dx-A"]) == 2
    assert result["raw_diagnoses"] == ["Dx-A", "Dx-B"]


@pytest.mark.parametrize("answer,expected", [("yes", ["Dx-A"]), ("no", ["Dx-Other"])])
def test_catchall_only_if_owning_tree_produces_no_specific_diagnosis(answer, expected):
    data = bundle_data(tree("T", {
        "fallback": node("fallback", yes={"fallback_diagnoses": ["Dx-Other"], "next": ["specific"]}),
        "specific": node("specific", yes={"diagnoses": ["Dx-A"]}),
    }))
    result = runner(data, Scripted("yes", "yes", answer)).run("case", case_id="c")
    assert result["raw_diagnoses"] == expected


def test_referral_executes_referenced_tree_even_if_its_screen_was_negative():
    data = bundle_data(tree("T1", {"a": node("a", yes={"referrals": ["T2"]})}),
                       tree("T2", {"b": node("b", yes={"diagnoses": ["Dx-A"]})}))
    result = runner(data, Scripted("yes", "no", "yes", "yes")).run("case", case_id="c")
    assert [(r["tree_id"], r["node_id"]) for r in result["tree_trace"]] == [("T1", "a"), ("T2", "b")]


def test_all_pairs_evaluated_from_original_set_even_after_an_elimination():
    data = bundle_data(tree("T", {"q": node("q", yes={"diagnoses": ["Dx-A", "Dx-B", "Dx-C"]})}))
    api = Scripted("yes", "yes", '["Dx-A"]', '["Dx-A","Dx-C"]', '["Dx-C"]')
    result = runner(data, api).run("case", case_id="c")
    assert [c["pair"] for c in result["pairwise_comparisons"]] == [
        ["Dx-A", "Dx-B"], ["Dx-A", "Dx-C"], ["Dx-B", "Dx-C"]]
    assert result["raw_diagnoses"] == ["Dx-A", "Dx-C"]


def test_cyclic_pairwise_elimination_is_reported_without_inventing_a_winner():
    data = bundle_data(tree("T", {"q": node("q", yes={"diagnoses": ["Dx-A", "Dx-B", "Dx-C"]})}))
    result = runner(data, Scripted("yes", "yes", '["Dx-A"]', '["Dx-C"]', '["Dx-B"]')).run("case", case_id="c")
    assert result["raw_diagnoses"] == [] and result["pairwise_all_eliminated"]


@pytest.mark.parametrize("invalid", ['[]', '["Dx-C"]', '["Dx-A", "Dx-A"]', '{"diagnoses":["Dx-A"]}'])
def test_invalid_pairwise_response_retries_the_same_pair(invalid):
    data = bundle_data(tree("T", {"q": node("q", yes={"diagnoses": ["Dx-A", "Dx-B"]})}))
    api = Scripted("yes", "yes", invalid, '["Dx-A"]')
    result = runner(data, api).run("case", case_id="c")
    assert result["raw_diagnoses"] == ["Dx-A"] and len(result["pairwise_comparisons"]) == 1
    assert api.requests[-1] == api.requests[-2]


def test_raw_tree_arm_is_separate_from_refined_paper_arm_and_base():
    data = bundle_data(tree("T", {"q": node("q")}), stage="raw")
    api = Scripted('{"diagnoses":[]}')
    engine = runner(data, api)
    with pytest.raises(ValueError, match="refined"):
        engine.run("case", case_id="c")
    assert not api.requests
    base = engine.run("case", case_id="c", method="base")
    assert base["status"] == "success" and base["raw_diagnoses"] == []
    assert base["screening"] == []
    assert runner(data, Scripted("no")).run("case", case_id="c", method="dt_raw")["status"] == "success"


def test_different_case_runs_never_reuse_previous_questions_or_answers():
    data = bundle_data(tree("T", {"q": node("q", yes={"diagnoses": ["Dx-A"]})}))
    api = Scripted("yes", "yes", "no")
    engine = runner(data, api)
    assert engine.run("first-case-text", case_id="first")["raw_diagnoses"] == ["Dx-A"]
    assert engine.run("second-case-text", case_id="second")["raw_diagnoses"] == []
    assert "first-case-text" not in api.requests[-1].messages[1]["content"]


@pytest.mark.parametrize("mutation", ["pending", "unknown_node", "unknown_referral", "cycle", "unreachable",
                                    "unknown_diagnosis", "duplicate_tree", "no_source", "no_refinement"])
def test_invalid_or_unreviewed_bundles_fail_before_any_model_call(mutation):
    data = bundle_data(tree("T", {"q": node("q")}))
    n = data["trees"][0]["nodes"]["q"]
    if mutation == "pending": data["review_status"] = "awaiting_review"
    if mutation == "unknown_node": n["yes"] = {"next": ["missing"]}
    if mutation == "unknown_referral": n["yes"] = {"referrals": ["missing"]}
    if mutation == "cycle": n["yes"] = {"next": ["q"]}
    if mutation == "unreachable": data["trees"][0]["nodes"]["unreachable"] = node("unreachable")
    if mutation == "unknown_diagnosis": n["yes"] = {"diagnoses": ["unlisted"]}
    if mutation == "duplicate_tree": data["trees"].append(copy.deepcopy(data["trees"][0]))
    if mutation == "no_source": n["source"]["pdf_pages"] = []
    if mutation == "no_refinement": data["refinements"] = []
    with pytest.raises(ValueError): Bundle.parse(data)


def test_cross_tree_cycle_is_rejected():
    with pytest.raises(ValueError, match="cycle"):
        Bundle.parse(bundle_data(tree("A", {"a": node("a", yes={"referrals": ["B"]})}),
                                 tree("B", {"b": node("b", yes={"referrals": ["A"]})})))


def test_duplicate_json_node_is_not_silently_overwritten(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"id":"one","id":"two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON"):
        load_bundle(path)


def test_actual_source_catalog_is_not_misrepresented_as_an_executable_tree():
    path = Path(__file__).resolve().parents[1] / "knowledge/paper_dtree/first2014_source_catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["book_tree_count"] == len(data["trees"]) == 29
    assert data["trees"][2]["print_start"] == 30 and data["trees"][2]["pdf_start"] == 47
    with pytest.raises(ValueError): load_bundle(path)


def test_paper_package_import_and_fixture_execution_do_not_import_project_layers(monkeypatch):
    real_import = builtins.__import__

    def independent_import(name, *args, **kwargs):
        if name.split(".")[0] in {"layers", "shared"}:
            raise AssertionError(f"Independent package imported project dependency {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", independent_import)
    import paper_dtree.engine
    importlib.reload(paper_dtree.engine)
    data = bundle_data(tree("T", {"q": node("q")}))
    assert runner(data, Scripted("no")).run("case", case_id="c")["status"] == "success"


def test_l3_port_is_detached_even_with_explicit_knowledge_and_never_falls_back(monkeypatch):
    from layers.layer3.llm import client
    from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider

    def forbidden(*args, **kwargs):
        pytest.fail("Default L3 must not construct the legacy backend or call any model")

    monkeypatch.setattr(client, "LegacyDSM5TreeReasoner", forbidden)
    monkeypatch.setattr(client, "ask_llm_stream", forbidden)
    with pytest.raises(client.CLINICRModelUnavailable, match="旧决策树已解耦"):
        client.APICLINICRReasoner(knowledge=JsonDiagnosticKnowledgeProvider()).generate_candidates({})
    assert not hasattr(client.APICLINICRReasoner, "_evaluate_tree_nodes")


def test_l3_port_only_delegates_when_an_explicit_project_compatible_generator_is_supplied():
    from layers.layer3.llm import client
    from tests.tree_test_support import controlled_generation

    expected, received = controlled_generation([]), []

    class ExplicitGenerator:
        def generate_candidates(self, profile):
            received.append(profile)
            return expected

    port = client.APICLINICRReasoner(candidate_generator=ExplicitGenerator())
    assert port.generate_candidates({"fixture": True}) == expected
    assert received == [{"fixture": True}]


def test_sdk_adapter_uses_the_explicit_settings_without_sdk_retries_or_json_mode(monkeypatch):
    from types import SimpleNamespace
    import openai
    from paper_dtree.models import Request
    from paper_dtree.transport import OpenAICompatibleTransport

    calls = []

    class FakeClient:
        def __init__(self, **options):
            calls.append(options)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **options):
            calls.append(options)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="yes"))],
                                   usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
                                   model="snapshot", id="request-1", system_fingerprint="fingerprint")

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    api = OpenAICompatibleTransport(api_key="fixture-key", base_url="https://fixture.invalid/v1")
    response = api.complete(Request("snapshot", ({"role": "user", "content": "fixture"},)))
    assert calls[0]["max_retries"] == 0
    assert (calls[1]["temperature"], calls[1]["top_p"], calls[1]["frequency_penalty"], calls[1]["presence_penalty"]) == (0, 1, 0, 0)
    assert "response_format" not in calls[1] and response.input_tokens == 3
