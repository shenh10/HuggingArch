"""A tiny live dashboard for the warm timing JSONL.

Run it beside the warm:

    python -m tools.warm_progress_ui --timing-file /tmp/warm.jsonl

The page auto-refreshes and reads only newly appended lines, so a long warm
does not make the browser reparse the whole file every second.
"""
from __future__ import annotations

import argparse
import json
import ssl
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "tools" / "warm_progress_ui.html"


class TimingState:
    """The timing file as append-only records, read incrementally."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._offset = 0
        self._size = 0

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            try:
                stat = self.path.stat()
            except OSError:
                return self._records
            if stat.st_size == self._size:
                return self._records
            if stat.st_size < self._offset:
                self._records.clear()
                self._offset = 0
            with self.path.open("rb") as stream:
                stream.seek(self._offset)
                data = stream.read()
            if not data:
                self._size = stat.st_size
                return self._records
            complete = data if data.endswith(b"\n") else (
                data[:data.rfind(b"\n") + 1] if b"\n" in data else b"")
            if complete:
                for line in complete.splitlines():
                    try:
                        self._records.append(json.loads(line))
                    except (UnicodeDecodeError, ValueError):
                        pass
                self._offset += len(complete)
            self._size = stat.st_size
            return self._records


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The page's whole model: counts, times and progress per GPU."""
    starts = {r["gpu"]: r for r in records if r["event"] == "gpu_start"}
    ends = {r["gpu"]: r for r in records if r["event"] == "gpu_end"}
    warm_start = next((r for r in records if r["event"] == "warm_start"), None)
    warm_end = next((r for r in records if r["event"] == "warm_end"), None)
    by_gpu: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "reads": 0, "read_s": 0.0,
        "layouts": 0, "layout_s": 0.0,
        "layout_computes": 0, "layout_compute_s": 0.0,
        "models": 0,
        "projections": 0, "projection_s": 0.0,
        "sweeps": 0, "sweep_s": 0.0,
        "read_times": [], "layout_times": [], "layout_compute_times": [],
        "projection_times": [], "sweep_times": [],
        "first_time": 0.0, "last_time": 0.0,
        "work_start": None,
    })
    by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "gpus": set(), "layouts": 0, "layout_s": 0.0,
        "layout_compute_s": 0.0, "projections": 0, "projection_s": 0.0,
        "sweeps": 0, "sweep_s": 0.0,
    })
    layouts = []

    for record in records:
        event = record["event"]
        gpu = record.get("gpu")
        model = record.get("model") or record.get("model_id")
        elapsed = float(record.get("elapsed_s", 0.0))
        timestamp = float(record.get("time", 0.0))
        if gpu is not None:
            row = by_gpu[gpu]
            row["first_time"] = row["first_time"] or timestamp
            row["last_time"] = max(row["last_time"], timestamp)
        if event == "read":
            row["reads"] += 1
            row["read_s"] += elapsed
            row["read_times"].append(elapsed)
        elif event == "layout":
            row["layouts"] += 1
            row["layout_s"] += elapsed
            row["layout_times"].append(elapsed)
            layouts.append(record)
            if model:
                by_model[model]["layouts"] += 1
                by_model[model]["layout_s"] += elapsed
        elif event == "layout_compute":
            row["layout_computes"] += 1
            row["layout_compute_s"] += elapsed
            row["layout_compute_times"].append(elapsed)
            # `time` is when a job finished and `elapsed_s` is how long it
            # ran. Their difference is when it actually started, unlike
            # `gpu_start`, which can precede a long wait for workers.
            started = timestamp - elapsed
            if row["work_start"] is None or started < row["work_start"]:
                row["work_start"] = started
            if model:
                by_model[model]["layout_compute_s"] += elapsed
        elif event == "model":
            row["models"] += 1
        elif event == "projection":
            row["projections"] += 1
            row["projection_s"] += elapsed
            row["projection_times"].append(elapsed)
            if model:
                by_model[model]["projections"] += 1
                by_model[model]["projection_s"] += elapsed
        elif event == "sweep":
            row["sweeps"] += 1
            row["sweep_s"] += elapsed
            row["sweep_times"].append(elapsed)
            if model:
                by_model[model]["sweeps"] += 1
                by_model[model]["sweep_s"] += elapsed
        if model and gpu is not None:
            by_model[model]["gpus"].add(gpu)

    gpu_rows = []
    for gpu, row in sorted(by_gpu.items()):
        start = starts.get(gpu)
        end = ends.get(gpu)
        started = float(start.get("time", 0.0)) if start else row["first_time"]
        finished = float(end.get("time", 0.0)) if end else row["last_time"]
        total_layouts = int(start.get("layouts", 0)) if start else 0
        total_models = int(start.get("models", 0)) if start else 0
        elapsed = max(0.0, finished - started)
        # ETA is measured over the interval in which layout jobs actually
        # ran. A class chain created beside another can wait for workers;
        # counting that wait as layout time makes its ETA look hours long.
        work_start = row["work_start"]
        active_elapsed = (max(0.0, finished - work_start)
                          if work_start is not None else 0.0)
        layout_rate = (row["layouts"] / active_elapsed
                       if active_elapsed else 0.0)
        eta = ((total_layouts - row["layouts"]) / layout_rate
               if total_layouts and layout_rate else None)
        gpu_rows.append({
            "gpu": gpu,
            "total_models": total_models,
            "total_layouts": total_layouts,
            "elapsed_s": elapsed,
            "active_elapsed_s": active_elapsed,
            "eta_s": eta,
            **{key: value for key, value in row.items()
               if not key.endswith("_times")},
            "read_p50_s": percentile(row["read_times"], .5),
            "read_p95_s": percentile(row["read_times"], .95),
            "layout_p50_s": percentile(row["layout_times"], .5),
            "layout_p95_s": percentile(row["layout_times"], .95),
            "layout_compute_p50_s": percentile(
                row["layout_compute_times"], .5),
            "layout_compute_p95_s": percentile(
                row["layout_compute_times"], .95),
            "projection_p50_s": percentile(row["projection_times"], .5),
            "projection_p95_s": percentile(row["projection_times"], .95),
            "sweep_p50_s": percentile(row["sweep_times"], .5),
            "sweep_p95_s": percentile(row["sweep_times"], .95),
            "done": bool(end),
        })

    top_models = sorted(
        ({
            "model": model,
            **{key: value for key, value in row.items()
               if key != "gpus"},
            "gpu_count": len(row["gpus"]),
        } for model, row in by_model.items()),
        key=lambda row: row["layout_compute_s"], reverse=True)[:12]
    top_layouts = sorted(
        layouts, key=lambda row: row.get("elapsed_s", 0.0), reverse=True)[:12]

    read_total = sum(row["read_s"] for row in by_gpu.values())
    layout_total = sum(row["layout_compute_s"] for row in by_gpu.values())
    projection_total = sum(row["projection_s"] for row in by_gpu.values())
    sweep_total = sum(row["sweep_s"] for row in by_gpu.values())
    return {
        "warm_start": warm_start,
        "warm_end": warm_end,
        "event_count": len(records),
        "overall": {
            "read_s": read_total,
            "layout_compute_s": layout_total,
            "projection_s": projection_total,
            "sweep_s": sweep_total,
            "reads": sum(row["reads"] for row in by_gpu.values()),
            "layouts": sum(row["layouts"] for row in by_gpu.values()),
            "models": sum(row["models"] for row in by_gpu.values()),
            "projections": sum(
                row["projections"] for row in by_gpu.values()),
            "sweeps": sum(row["sweeps"] for row in by_gpu.values()),
        },
        "gpus": gpu_rows,
        "top_models": top_models,
        "top_layouts": top_layouts,
        "recent_events": records[-20:],
    }


class Handler(BaseHTTPRequestHandler):
    state: TimingState

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            body = HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif route == "/api/progress":
            body = json.dumps(
                summarize(self.state.snapshot()),
                separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *_args: Any) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-file", default=str(
        Path("/tmp/warm_full_timing.jsonl")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ssl-certfile")
    parser.add_argument("--ssl-keyfile")
    args = parser.parse_args()
    state = TimingState(Path(args.timing_file))
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            parser.error(
                "--ssl-certfile and --ssl-keyfile must be given together")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.ssl_certfile, args.ssl_keyfile)
        server.socket = context.wrap_socket(
            server.socket, server_side=True)
        scheme = "https"
    else:
        scheme = "http"
    print(f"warm progress UI: {scheme}://{args.host}:{args.port}")
    print(f"timing file: {args.timing_file}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
