"""Opt-in research runner utilities; never installed on the production API path."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import threading
import time
import uuid
from typing import Any

import httpx
from openai import APIConnectionError, APIError, APIStatusError, OpenAI

try:
    import httpx2
except ImportError:
    httpx2 = None

TRANSPORT_ERRORS = (httpx.TransportError,) + ((httpx2.TransportError,) if httpx2 else ())
TIMEOUT_ERRORS = (httpx.TimeoutException, TimeoutError) + ((httpx2.TimeoutException,) if httpx2 else ())


def legacy_stream_retryable(record):
    """Recognize the observed, misclassified HTTP-200 stream failure on recovery."""
    return record.get("status_code") == 200 and bool(
        {"httpx2.RemoteProtocolError", "httpcore2.RemoteProtocolError"}
        & set(record.get("exception_types") or []))

from layers.layer3.diagnosis.g1_cache import content_hash
from layers.layer3.diagnosis.schemas import TreeNodeJudgment
from layers.layer3.llm.observability import emit, update_context


class CaseBudgetExceeded(RuntimeError):
    pass


class BatchBudgetExceeded(RuntimeError):
    pass


class NonRetryableProviderError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"Non-retryable provider HTTP {status_code}")
        self.status_code = status_code


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def usage_values(usage):
    if not usage:
        return None
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int) or min(prompt, completion) < 0:
        return None
    details = usage.get("prompt_tokens_details") or {}
    hit = usage.get("prompt_cache_hit_tokens", details.get("cached_tokens"))
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return {"prompt": prompt, "completion": completion, "cached": hit,
            "reasoning": reasoning, "reference_all_miss_peak_yuan": (2 * prompt + 8 * completion) / 1e6}


def reasoning_observation(entry):
    usage = entry.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    explicit = details.get("reasoning_tokens")
    if isinstance(explicit, int):
        return {"raw_reasoning_tokens": explicit, "effective_reasoning_tokens": explicit, "basis": "provider_explicit"}
    text, completion = details.get("text_tokens"), usage.get("completion_tokens")
    if isinstance(text, int) and text == completion and not entry.get("reasoning_stream_seen"):
        return {"raw_reasoning_tokens": explicit, "effective_reasoning_tokens": 0,
                "basis": "derived: text_tokens equals all completion_tokens and no reasoning stream; raw null retained"}
    return {"raw_reasoning_tokens": explicit, "effective_reasoning_tokens": None, "basis": "not_exposed_or_not_reconcilable"}


class ReferenceBudget:
    """Per-case and shared limits; concurrent reservations wait, rather than fail cases."""
    def __init__(self, *, batch_limit, case_limit):
        self.batch_limit, self.case_limit = batch_limit, case_limit
        self.spent, self.reserved, self.unknown_upper = 0.0, 0.0, 0.0
        self.by_case: dict[str, float] = {}
        self.condition = threading.Condition()

    def reserve(self, case_id, prompt, max_tokens):
        # One HTTP attempt: SDK retries are explicitly disabled.
        amount = ((len(prompt.encode("utf-8")) + 128) * 2 + max_tokens * 8) / 1e6
        with self.condition:
            while True:
                if self.by_case.get(case_id, 0) + amount > self.case_limit:
                    raise CaseBudgetExceeded("Per-case reference limit; other cases may continue")
                if self.spent + amount > self.batch_limit:
                    raise BatchBudgetExceeded("Batch reference limit cannot fund another request")
                if self.spent + self.reserved + amount <= self.batch_limit:
                    self.reserved += amount
                    return amount
                self.condition.wait(timeout=.25)

    def finish(self, case_id, amount, usage):
        values = usage_values(usage)
        charged = values["reference_all_miss_peak_yuan"] if values else amount
        with self.condition:
            self.reserved = max(0.0, self.reserved - amount)
            self.spent += charged
            self.by_case[case_id] = self.by_case.get(case_id, 0) + charged
            if values is None:
                self.unknown_upper += amount
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"reference_spent_upper_yuan": self.spent, "reference_reserved_yuan": self.reserved,
                    "unknown_usage_upper_yuan": self.unknown_upper, "per_case_reference": dict(self.by_case),
                    "batch_limit": self.batch_limit, "case_limit": self.case_limit,
                    "actual_platform_invoice": None}


def retryable(error):
    # Some streaming transports surface a wrapper instead of httpx's exception.
    # Accept typed causes or an explicit transient provider code, never messages.
    seen, provider_internal = set(), False
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, APIStatusError):
            return error.status_code in {408, 429} or error.status_code >= 500
        if isinstance(error, APIError) and error.code == "internal_server_error":
            # A streaming error envelope can arrive after an HTTP 200 header.
            # Still inspect causes so an explicit HTTP 4xx takes precedence.
            provider_internal = True
        if isinstance(error, (APIConnectionError, TimeoutError) + TRANSPORT_ERRORS):
            return True
        error = error.__cause__ or error.__context__
    return provider_internal


def exception_types(error):
    seen, result = set(), []
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        result.append(f"{type(error).__module__}.{type(error).__qualname__}")
        error = error.__cause__ or error.__context__
    return result


def error_category(error):
    seen, fallback = set(), "provider_stream_or_client_error"
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, APIStatusError):
            if error.status_code == 429:
                return "rate_limit"
            if error.status_code == 408:
                return "timeout"
            return "server_error" if error.status_code >= 500 else "http_client_error"
        if isinstance(error, APIError) and error.code == "internal_server_error":
            fallback = "provider_internal_error"
        if isinstance(error, TIMEOUT_ERRORS):
            return "timeout"
        if isinstance(error, ConnectionResetError):
            return "connection_reset"
        if isinstance(error, (APIConnectionError,) + TRANSPORT_ERRORS):
            fallback = "connection_error"
            # Preserve a more specific socket cause when available.
            cause = error.__cause__ or error.__context__
            if cause is None:
                return "connection_error"
        error = error.__cause__ or error.__context__
    return fallback


class ReliableStreaming:
    """Retry whole HTTP generations; discard partial text before a retry."""
    def __init__(self, *, api_key, base_url, journal, budget, max_attempts=3, read_timeout=300,
                 max_tokens=16000, thinking=None, client_factory=OpenAI, sleep=time.sleep):
        self.api_key, self.base_url = api_key, base_url
        self.journal, self.budget = journal, budget
        self.max_attempts, self.read_timeout, self.max_tokens = max_attempts, read_timeout, max_tokens
        self.thinking, self.client_factory, self.sleep = thinking, client_factory, sleep
        self.local = threading.local()
        self.lock = threading.Lock()
        self.rng = random.Random(20261015)

    def bind(self, case_id, requests, *, thinking="inherit", arm=None):
        self.local.case_id, self.local.requests, self.local.batch = case_id, requests, 0
        self.local.thinking = self.thinking if thinking == "inherit" else thinking
        self.local.arm = arm
        self.local.seen_batches = {}
        self.local.tags = {"stage": "verifier", "node_ids": []}

    def tags(self, stage, node_ids):
        self.local.tags = {"stage": stage, "node_ids": node_ids}
        update_context(stage=stage, node_ids=node_ids,
                       scope="node" if node_ids else "verifier" if stage == "verifier" else "case",
                       attempt_id=None, request_id=None, finish_reason=None)

    def ask(self, prompt, model=None, temperature=0, *, timeout=None, extra_body=None, response_format=None):
        self.local.batch += 1
        batch = self.local.batch
        batch_key = (self.local.tags["stage"], tuple(self.local.tags["node_ids"]))
        structured_attempt = self.local.seen_batches.get(batch_key, 0) + 1
        self.local.seen_batches[batch_key] = structured_attempt
        body = dict(extra_body or {})
        thinking = getattr(self.local, "thinking", self.thinking)
        if thinking is not None:
            body["enable_thinking"] = thinking
        for attempt in range(1, self.max_attempts + 1):
            amount = self.budget.reserve(self.local.case_id, prompt, self.max_tokens)
            entry = {"case_id": self.local.case_id, **self.local.tags, "batch_number": batch,
                     "attempt_id": uuid.uuid4().hex,
                     "arm": self.local.arm,
                     "structured_attempt": structured_attempt, "is_structured_retry": structured_attempt > 1,
                     "started_at": now(),
                     "attempt": attempt + self.local.tags.get("prior_http_attempts", 0), "requested_model": model, "temperature": temperature,
                     "response_format": response_format, "extra_body": body,
                     "read_timeout_seconds": self.read_timeout, "sdk_max_retries": 0, "stream": True,
                     "prompt_characters": len(prompt), "usage": None, "status_code": None,
                     "raw_output": "", "retryable": False, "reasoning_stream_seen": False}
            entry["reference_reservation_upper_yuan"] = amount
            update_context(stage=entry["stage"], node_ids=entry["node_ids"],
                           attempt_id=entry["attempt_id"], request_id=None, finish_reason=None)
            with self.lock:
                self.journal.write(json.dumps({**entry, "event": "attempt_started"}, ensure_ascii=False) + "\n")
                self.journal.flush()
            emit("attempt_started", **entry)
            self.local.requests.append(entry)
            started, stream, api = time.perf_counter(), None, None
            parts, error = [], None
            try:
                api = self.client_factory(api_key=self.api_key, base_url=self.base_url, max_retries=0,
                                          timeout=httpx.Timeout(self.read_timeout, connect=15, write=60, pool=60))
                request = {"model": model, "messages": [{"role": "user", "content": prompt}],
                           "temperature": temperature, "stream": True, "max_tokens": self.max_tokens,
                           "stream_options": {"include_usage": True}}
                if body:
                    request["extra_body"] = body
                if response_format:
                    request["response_format"] = response_format
                stream = api.chat.completions.create(**request)
                entry["status_code"] = 200
                response = getattr(stream, "response", None)
                headers = getattr(response, "headers", {})
                entry["response_request_id"] = headers.get("x-request-id") or headers.get("request-id")
                for chunk in stream:
                    entry.setdefault("time_to_first_chunk_seconds", time.perf_counter() - started)
                    entry["response_model"] = chunk.model
                    entry["completion_id"] = getattr(chunk, "id", None)
                    if getattr(chunk, "system_fingerprint", None):
                        entry["system_fingerprint"] = chunk.system_fingerprint
                    if getattr(chunk, "usage", None):
                        entry["usage"] = chunk.usage.model_dump(mode="json")
                    for choice in chunk.choices or []:
                        if getattr(choice.delta, "reasoning_content", None):
                            entry["reasoning_stream_seen"] = True
                        if choice.delta.content:
                            entry.setdefault("time_to_first_content_seconds", time.perf_counter() - started)
                            parts.append(choice.delta.content)
                        if choice.finish_reason:
                            entry["finish_reason"] = choice.finish_reason
            except Exception as exc:
                error = exc
                category, can_retry = error_category(exc), retryable(exc)
                entry.update(error_type=type(exc).__name__, status_code=getattr(exc, "status_code", entry["status_code"]),
                             provider_error_code=getattr(exc, "code", None),
                             error_category=category,
                             exception_types=exception_types(exc), retryable=can_retry,
                             failure_origin="provider_stream" if category == "provider_internal_error" else
                                 "client_transport" if not isinstance(exc, APIStatusError) and can_retry else "http_or_client")
            finally:
                if stream is not None and hasattr(stream, "close"):
                    stream.close()
                if api is not None:
                    api.close()
                entry.update(raw_output="".join(parts), completed_at=now(), elapsed_seconds=round(time.perf_counter() - started, 3),
                             usage_values=usage_values(entry["usage"]))
                update_context(request_id=entry.get("response_request_id") or entry.get("completion_id"),
                               finish_reason=entry.get("finish_reason"))
                if error is not None and entry["retryable"] and attempt < self.max_attempts:
                    with self.lock:
                        entry["backoff_seconds"] = min(30, 2 ** (attempt - 1)) + self.rng.random()
                self.budget.finish(self.local.case_id, amount, entry["usage"])
                with self.lock:
                    self.journal.write(json.dumps({**entry, "event": "attempt_finished"}, ensure_ascii=False) + "\n")
                    self.journal.flush()
                emit("attempt_finished", **{k: v for k, v in entry.items() if k != "raw_output"})
            if error is None:
                return "".join(parts)
            if not entry["retryable"] or attempt == self.max_attempts:
                if isinstance(error, APIStatusError) and 400 <= error.status_code < 500 and not entry["retryable"]:
                    raise NonRetryableProviderError(error.status_code) from error
                raise error
            self.sleep(entry["backoff_seconds"])
        raise AssertionError("Unreachable retry state")


def recover_ledger(path):
    """Include started-but-unfinished attempts at their recorded maximum, never zero."""
    path = Path(path)
    entries = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            entries[item["attempt_id"]] = item
    spent, unknown, by_case = 0.0, 0.0, {}
    for item in entries.values():
        values = usage_values(item.get("usage"))
        amount = values["reference_all_miss_peak_yuan"] if values else item["reference_reservation_upper_yuan"]
        spent += amount
        by_case[item["case_id"]] = by_case.get(item["case_id"], 0) + amount
        if not values:
            unknown += amount
    return {"spent": spent, "unknown_upper": unknown, "by_case": by_case,
            "unfinished_http_attempts": sum(i["event"] != "attempt_finished" for i in entries.values())}


class NodeCheckpoint:
    """Validated judgments only; transport policy is provenance, not a clinical input."""
    def __init__(self, root, context):
        self.context = context
        self.path = Path(root) / (content_hash(context) + ".json")
        self.nodes = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data["context"] != context or data["nodes_hash"] != content_hash(data["nodes"]):
                raise ValueError("Checkpoint context or integrity mismatch")
            self.nodes = data["nodes"]

    def get(self, node):
        cached = self.nodes.get(node["node_id"])
        if cached is None:
            return None
        if cached["definition"] != node:
            raise ValueError("Checkpoint node definition changed")
        return TreeNodeJudgment.model_validate(cached["judgment"])

    def save(self, nodes, judgments, *, origin):
        if {n["node_id"] for n in nodes} != {j.node_id for j in judgments} or len(nodes) != len(judgments):
            raise ValueError("Only complete validated batches may be checkpointed")
        for node, judgment in zip(nodes, sorted(judgments, key=lambda j: [n["node_id"] for n in nodes].index(j.node_id))):
            entry = {"definition": node, "judgment": judgment.model_dump(mode="json"), "origin": origin}
            old = self.nodes.get(node["node_id"])
            if old is not None and (old["definition"] != node or old["judgment"] != entry["judgment"]):
                raise ValueError("Never replace an already completed node")
            self.nodes.setdefault(node["node_id"], entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"context": self.context, "nodes": self.nodes,
                                        "nodes_hash": content_hash(self.nodes)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


def attach_checkpoints(reasoner, checkpoint, transport):
    evaluate = reasoner._evaluate_tree_nodes
    stats = {"reused_nodes": 0, "new_nodes": 0}
    def cached(profile, nodes):
        answers = []
        for batch in reasoner.node_batches(nodes):
            loaded = {n["node_id"]: checkpoint.get(n) for n in batch}
            missing = [n for n in batch if loaded[n["node_id"]] is None]
            stats["reused_nodes"] += len(batch) - len(missing)
            if missing:
                stage = {"criterion": "criteria", "exclusion": "guard"}.get(missing[0]["stage"], missing[0]["stage"])
                transport.tags(stage, [n["node_id"] for n in missing])
                fresh = evaluate(profile, missing)
                checkpoint.save(missing, fresh, origin="repair_live_validated")
                stats["new_nodes"] += len(fresh)
                loaded.update({j.node_id: j for j in fresh})
            answers.extend(loaded[n["node_id"]] for n in batch)
        transport.tags("verifier", [])
        return answers
    reasoner._evaluate_tree_nodes = cached
    return stats
