"""Local, read-only progress view. Never imports or invokes the diagnosis engine."""
import argparse
import ctypes
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent


def read_text(path):
    if sys.platform != "win32":
        return path.read_text(encoding="utf-8-sig")
    # Permit atomic replacements by the running worker while observing its files.
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong,
                                  ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    handle = kernel.CreateFileW(str(path), 0x80000000, 7, None, 3, 128, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    with os.fdopen(fd, "rb") as stream:
        return stream.read().decode("utf-8-sig")


def load_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(read_text(path))


def load_lines(path):
    if not path.exists():
        return []
    lines, result = read_text(path).splitlines(), []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A live append may be midway through its last line. Never edit logs.
            if index != len(lines) - 1:
                raise
    return result


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def process_alive(pid):
    if not pid or sys.platform != "win32":
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False if ctypes.get_last_error() == 87 else None
    try:
        code = ctypes.c_ulong()
        return code.value == 259 if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
    finally:
        kernel.CloseHandle(handle)


def age_seconds(timestamp, now):
    if not timestamp:
        return None
    return max(0, (now - datetime.fromisoformat(timestamp)).total_seconds())


def build_status(run, *, pid=None, alive=None):
    now = datetime.now(timezone(timedelta(hours=8)))
    manifest = load_json(run / "joint-manifest.json", {})
    state = load_json(run / "joint-state.json", {})
    final = load_json(run / "joint-summary.json")
    ended_at = final.get("finished_at") if final else None
    elapsed_at = datetime.fromisoformat(ended_at) if ended_at else now
    identity = manifest.get("identity", {})
    declared = load_json(run / "manifest.A0.json", {}).get("identity", {})
    ids = declared.get("case_ids", [])
    attempts, cells, arms = [], [], {}
    pending_reservation, consumed, unknown_usage, known_cost = 0, 0, 0, 0
    failures = {"transport": 0, "contract": 0, "budget": 0, "administrative": 0}
    for arm in ("A0", "A1"):
        ledger = load_lines(run / "runs" / arm / "requests.jsonl")
        by_attempt = {r["attempt_id"]: r for r in ledger}
        for raw in by_attempt.values():
            usage = raw.get("usage") or {}
            prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
            finished = raw.get("event") == "attempt_finished"
            reservation = raw.get("reference_reservation_upper_yuan")
            known = type(prompt) is int and type(completion) is int and min(prompt, completion) >= 0
            cost = (2 * prompt + 8 * completion) / 1e6 if known else reservation if finite(reservation) else None
            if finished:
                if cost is not None:
                    consumed += cost
                if known:
                    known_cost += cost
                else:
                    unknown_usage += 1
            elif finite(reservation):
                pending_reservation += reservation
            elapsed = raw.get("elapsed_seconds") if finished else age_seconds(raw.get("started_at"), now)
            attempts.append({"arm": arm, "case_id": raw.get("case_id"), "stage": raw.get("stage"),
                "node_ids": raw.get("node_ids", []), "started_at": raw.get("started_at"),
                "completed_at": raw.get("completed_at"), "pending": not finished,
                "attempt": raw.get("attempt"), "elapsed_seconds": elapsed,
                "status_code": raw.get("status_code"), "error_type": raw.get("error_type"),
                "error_category": raw.get("error_category"), "reference_yuan": cost if finished else None})
        counts = {"success": 0, "failed": 0, "administrative": 0, "active": 0, "waiting": 0}
        for case in ids:
            result = load_json(run / "runs" / arm / f"{case}.json")
            requests = [r for r in attempts if r["arm"] == arm and r["case_id"] == case]
            last = requests[-1] if requests else None
            if result is not None:
                failure = result.get("failure_class")
                status = "administrative" if failure == "administrative" else "failed" if result.get("run_failure") else "success"
                if failure:
                    failures[failure] = failures.get(failure, 0) + 1
            else:
                failure = None
                status = "active" if requests else "waiting"
            counts[status] += 1
            cells.append({"arm": arm, "case_id": case, "status": status, "failure_class": failure,
                          "stage": last["stage"] if last else None,
                          "http_attempts": len(requests), "last_request": last})
        arms[arm] = {**counts, "terminal": counts["success"] + counts["failed"] + counts["administrative"], "planned": len(ids)}
    attempts.sort(key=lambda r: r.get("started_at") or "")
    active = [r for r in attempts if r["pending"]]
    last = active[-1] if active else attempts[-1] if attempts else None
    joint_complete = sum(all(any(c["case_id"] == case and c["arm"] == arm and c["status"] == "success" for c in cells) for arm in arms) for case in ids)
    running = process_alive(pid) if alive is None else alive
    reasons = final.get("stop_reasons", []) if final else state.get("stop_reasons", [])
    if final is not None:
        status = "stopped" if reasons else "finished"
    elif running is False:
        status = "abnormal"
    elif reasons:
        status = "closing"
    else:
        status = "running" if running else "unconfirmed"
    return {"checked_at": now.isoformat(), "run_id": run.name, "status": status, "process_alive": running,
        "started_at": manifest.get("started_at"), "finished_at": ended_at,
        "elapsed_seconds": age_seconds(manifest.get("started_at"), elapsed_at),
        "model": identity.get("model"), "planned_cases": len(ids), "planned_arms": len(ids) * 2,
        "terminal_arms": sum(a["terminal"] for a in arms.values()), "joint_complete": joint_complete,
        "arms": arms, "cells": cells, "current": last, "requests": attempts[-8:][::-1],
        "http_attempts": len(attempts), "stop_reasons": reasons, "failures": failures,
        "reference_budget": {"consumed_upper_yuan": round(consumed, 6), "known_usage_yuan": round(known_cost, 6),
            "pending_reservation_yuan": round(pending_reservation, 6), "finished_usage_missing": unknown_usage,
            "limit_yuan": identity.get("budget_reference", {}).get("batch_limit_yuan", 30), "actual_bill": None},
        "final_ready": final is not None, "final": final,
        "ledger_ready": (run / "observability/joint.summary.md").exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run / "joint-manifest.json").is_file():
        parser.error("An existing joint-run manifest is required")
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            try:
                if route == "/":
                    body, mime, code = (HERE / "index.html").read_bytes(), "text/html; charset=utf-8", 200
                elif route == "/status.json":
                    body, mime, code = json.dumps(build_status(run, pid=args.pid), ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", 200
                elif route == "/results.json":
                    value = load_json(run / "joint-summary.json", {"ready": False})
                    body, mime, code = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"), "application/json; charset=utf-8", 200
                elif route == "/ledger":
                    path = run / "observability/joint.summary.md"
                    body, mime, code = (path.read_bytes() if path.exists() else "联合账本尚未生成".encode("utf-8")), "text/plain; charset=utf-8", 200
                else:
                    body, mime, code = b"Not found", "text/plain", 404
            except (OSError, ValueError) as error:
                body, mime, code = json.dumps({"error": type(error).__name__}).encode(), "application/json", 503
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    info = {"url": f"http://127.0.0.1:{server.server_port}/", "run": str(run), "read_only": True,
            "diagnosis_pid": args.pid}
    (HERE / "server-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
