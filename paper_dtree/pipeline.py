"""Figure 1 is the outer seven-stage experiment; Figure 2 is inside stage 2."""
from __future__ import annotations

from . import prompts
from .postprocess import Catalog, ReconciliationStore, diagnosis_list, nonempty, reconcile
from .scoring import score_case


STAGES = ((1, "query"), (2, "model"), (3, "raw_response"), (4, "simplify"),
          (5, "fuzzy_match"), (6, "reconcile"), (7, "scoring"))


class SevenStagePipeline:
    def __init__(self, catalog: Catalog, store: ReconciliationStore, *, runner=None, record_stage=None):
        # Fail before spending a model request if the paper's matching dependency is absent.
        from rapidfuzz.fuzz import WRatio  # noqa: F401
        self.catalog, self.store, self.runner = catalog, store, runner
        self.record_stage = record_stage or (lambda stage: None)
        if runner is not None and runner.bundle.standard != catalog.standard:
            raise ValueError("Tree and diagnostic catalog standards must match")

    def _stage(self, stages, number, status, payload):
        item = {"number": number, "name": dict(STAGES)[number], "status": status, "payload": payload}
        stages.append(item)
        self.record_stage(item)

    def run(self, case, *, case_id, method="dt", reference=None, reference_source=None):
        if self.runner is None:
            raise ValueError("Model runner required; use resume for saved model outputs")
        nonempty(case, "case vignette")
        nonempty(case_id, "case_id")
        self._check_reference(reference, reference_source)
        query = {"case_id": case_id, "vignette": case, "method": method,
                 "standard": self.catalog.standard, "model": self.runner.model,
                 "entry_profile_id": self.runner.entry_profile.id if self.runner.entry_profile else "bundle_tree_screening",
                 "system_prompt": prompts.system(self.catalog.standard),
                 "case_prompt": f"The clinical case is as follows: {case}"}
        stages, events = [], []
        self._stage(stages, 1, "completed", query)
        original_record = self.runner.record

        def record(event):
            events.append(event)
            original_record(event)

        self.runner.record = record
        try:
            # Reference labels never enter stage 1, model prompts, tree traversal or pairwise comparison.
            model_result = self.runner.run(case, case_id=case_id, method=method)
        except Exception as exc:
            self._stage(stages, 2, "failed", {"error_type": type(exc).__name__, "error": str(exc)})
            for number, _ in STAGES[2:]:
                self._stage(stages, number, "not_started", {"reason": "model_stage_failed"})
            return self._result(query, stages, "failed")
        finally:
            self.runner.record = original_record
        if model_result.get("status") in {"awaiting_tree_content", "awaiting_patient"}:
            waiting = model_result["status"]
            self._stage(stages, 2, waiting, model_result)
            for number, _ in STAGES[2:]:
                self._stage(stages, number, "not_started", {"reason": "selected_tree_content_unavailable" if waiting == "awaiting_tree_content" else "awaiting_patient_response"})
            return self._result(query, stages, waiting)
        responses = [{"stage": e["stage"], "attempt": e["attempt"], "response": e["response"]}
                     for e in events if e.get("event") == "request_attempt" and "response" in e]
        return self._finish(query, model_result, responses, stages, reference, reference_source)

    @staticmethod
    def _check_reference(reference, source):
        if reference is not None:
            diagnosis_list(reference)
            nonempty(source, "Author-designated reference source")

    def resume(self, saved, *, reference=None, reference_source=None):
        """Continue stages 4-7 with saved raw outputs; never make a model request."""
        self._check_reference(reference, reference_source)
        if saved.get("standard") != self.catalog.standard or saved.get("catalog_id") != self.catalog.id:
            raise ValueError("Resume must use the same standard and versioned diagnostic catalog")
        stages_by_id = {stage["number"]: stage for stage in saved["stages"]}
        if any(stages_by_id[n]["status"] != "completed" for n in (1, 2, 3)):
            raise ValueError("Cannot resume without completed saved query/model/raw-response stages")
        query = stages_by_id[1]["payload"]
        model_result = stages_by_id[2]["payload"]
        stages = []
        self._stage(stages, 1, "completed", query)
        responses = stages_by_id[3]["payload"]["responses"]
        if reference is None:
            old = stages_by_id.get(4, {}).get("payload", {}).get("author_reference")
            if old is not None:
                reference, reference_source = old["raw_diagnoses"], old["source_ref"]
        return self._finish(query, model_result, responses, stages, reference, reference_source)

    def _finish(self, query, model_result, responses, stages, reference, reference_source):
        if (model_result.get("status") != "success" or model_result.get("standard") != self.catalog.standard
                or model_result.get("case_id") != query["case_id"]):
            raise ValueError("Invalid model result for this query")
        raw = diagnosis_list(model_result["raw_diagnoses"])
        self._stage(stages, 2, "completed", model_result)
        self._stage(stages, 3, "completed", {"diagnoses": raw, "responses": responses})
        simple_pred = self.catalog.simplify(raw)
        simple_ref = self.catalog.simplify(reference) if reference is not None else None
        self._stage(stages, 4, "completed", {
            "predictions": simple_pred, "reference": simple_ref,
            "author_reference": {"raw_diagnoses": reference, "source_ref": reference_source}
            if reference is not None else None})
        fuzzy_pred = self.catalog.fuzzy_match(simple_pred)
        fuzzy_ref = self.catalog.fuzzy_match(simple_ref) if simple_ref is not None else None
        self._stage(stages, 5, "completed", {"predictions": fuzzy_pred, "reference": fuzzy_ref})
        pred = reconcile(self.catalog, fuzzy_pred, self.store)
        ref = reconcile(self.catalog, fuzzy_ref, self.store) if fuzzy_ref is not None else None
        pending = pred["status"] != "completed" or (ref is not None and ref["status"] != "completed")
        status = "awaiting_reconciliation" if pending else "completed"
        self._stage(stages, 6, status, {"predictions": pred, "reference": ref})
        if pending:
            self._stage(stages, 7, "not_started", {"reason": "unresolved_reconciliation"})
            return self._result(query, stages, "awaiting_reconciliation")
        if ref is None:
            self._stage(stages, 7, "awaiting_reference", {"reason": "author_designated_reference_not_supplied"})
            return self._result(query, stages, "awaiting_reference")
        self._stage(stages, 7, "completed", score_case(pred, ref))
        return self._result(query, stages, "completed")

    def _result(self, query, stages, status):
        result = {"case_id": query["case_id"], "method": query["method"], "status": status,
                  "standard": self.catalog.standard, "catalog_id": self.catalog.id,
                  "catalog_source": self.catalog.source_ref, "stages": stages}
        if stages[-1]["number"] == 7 and stages[-1]["status"] == "completed":
            result["scoring"] = stages[-1]["payload"]
        return result
