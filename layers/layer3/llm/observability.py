"""Opt-in research events; no prompt or judgment changes when collecting logs."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import threading

_SINK = ContextVar("l3_research_event_sink", default=None)
_CONTEXT = ContextVar("l3_research_event_context", default=None)
EVENT_VERSION = "l3-research-events-v1"
ENVELOPE_VERSION = "l3-event-envelope-v1"


def current_context():
    return dict(_CONTEXT.get() or {})


def update_context(**fields):
    context = _CONTEXT.get()
    if context is not None:
        context.update(fields)


def emit(event, **fields):
    sink = _SINK.get()
    if sink is not None:
        sink({"version": EVENT_VERSION, "event_version": EVENT_VERSION,
              **current_context(), **fields, "event": event})


@contextmanager
def capture_events(sink, **context):
    token = _SINK.set(sink)
    context_token = _CONTEXT.set(dict(context))
    try:
        yield
    finally:
        _CONTEXT.reset(context_token)
        _SINK.reset(token)


class EventJournal:
    """One append-only file sequence; existing records are never rewritten."""
    def __init__(self, path, *, run_id, arm):
        self.path, self.run_id, self.arm = Path(path), run_id, arm
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq, self.lock = 0, threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                value = json.loads(line)
                if value.get("seq") != self.seq + 1:
                    raise ValueError("Cannot append to an incomplete or legacy event sequence")
                self.seq += 1

    def __call__(self, value):
        with self.lock:
            self.seq += 1
            record = {"version": EVENT_VERSION, "event_version": EVENT_VERSION,
                      "envelope_version": ENVELOPE_VERSION,
                      "run_id": self.run_id, "arm": self.arm, "case_id": None,
                      "stage": "administrative", "scope": "case", "node_ids": [],
                      "attempt_id": None, "request_id": None, **value,
                      "seq": self.seq,
                      "ts": datetime.now(timezone(timedelta(hours=8))).isoformat()}
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
