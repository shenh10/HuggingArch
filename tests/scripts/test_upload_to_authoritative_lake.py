from __future__ import annotations

import pytest

import tools.upload_to_authoritative_lake as uploader
from tests.corpus_row import row, to_csv


class _Report:
    def ok(self) -> bool:
        return True


def _csv(run_ids: list[str]) -> str:
    return to_csv([row(run_id=run_id) for run_id in run_ids])


def _fake_lake(calls: list, row_count: int, private: set):
    """The real client with only its HTTP replaced: `private` is the runs the
    server holds private batches of (contribute 404s on any other)."""
    class Lake(uploader.AuthoritativeLake):
        def post(self, path, payload):
            calls.append((path, payload))
            if path == "/api/bench/upload":
                return {"upload_id": "22222222-2222-4222-8222-222222222222",
                        "row_count": row_count}
            if path.endswith("/contribute?within_peak=true"):
                run = path.split("/")[4]
                if run not in private:
                    raise RuntimeError("HTTP 404 from contribute: no private batch")
                private.discard(run)
                return {"batches": 1}
            if path == "/api/bench/promote":
                return {"promoted": [{"batches": 1}]}
            raise AssertionError(path)
    return Lake


def _run(monkeypatch, tmp_path, lake, run_ids, *extra):
    path = tmp_path / "results.csv"
    path.write_text(_csv(run_ids), encoding="utf-8")
    monkeypatch.setattr(uploader, "check_paths", lambda _: _Report())
    monkeypatch.setattr(uploader, "format_report", lambda _: "checked")
    monkeypatch.setattr(uploader, "AuthoritativeLake", lake)
    return uploader.main([
        "gpu", str(path), "--host", "https://example.invalid", "--token", "hapat_ok",
        "--owner", "11111111-1111-4111-8111-111111111111", *extra])


def test_a_run_whose_resend_adds_nothing_is_still_shared(monkeypatch, tmp_path):
    # The first send stored the rows and the process stopped before the
    # promote; the resend is deduplicated to 0 rows. The run's private batch
    # must still be nominated and promoted, or it stays private for good.
    calls: list = []
    rc = _run(monkeypatch, tmp_path, _fake_lake(calls, 0, {"run-a"}), ["run-a"])
    assert rc == 0
    assert [p for p, _ in calls] == [
        "/api/bench/upload",
        "/api/bench/run/run-a/contribute?within_peak=true",
        "/api/bench/promote",
    ]
    assert calls[-1][1] == {"sets": [{
        "owner": "11111111-1111-4111-8111-111111111111", "kind": "run", "id": "run-a"}]}


def test_each_run_is_shared_once_after_all_its_parts(monkeypatch, tmp_path):
    calls: list = []
    rc = _run(monkeypatch, tmp_path, _fake_lake(calls, 1, {"run-a", "run-b"}),
              ["run-a", "run-a", "run-a", "run-b"], "--chunk-rows", "2")
    assert rc == 0
    assert [p for p, _ in calls] == [
        "/api/bench/upload", "/api/bench/upload", "/api/bench/upload",
        "/api/bench/run/run-a/contribute?within_peak=true", "/api/bench/promote",
        "/api/bench/run/run-b/contribute?within_peak=true", "/api/bench/promote",
    ]


def test_the_local_service_is_not_accepted_as_authoritative(monkeypatch):
    monkeypatch.setattr(uploader, "_token", lambda _args: "hapat_ok")
    with pytest.raises(ValueError, match=r"not 127\.0\.0\.1:8100"):
        uploader.main(["gpu", "x.csv", "--host", "http://127.0.0.1:8100",
                       "--private"])


def test_a_large_run_goes_up_in_parts_no_request_over_the_limit(monkeypatch, tmp_path):
    # The public API's edge proxy cuts a request at 100 s (HTTP 524); one
    # 8k-row upload already exceeded it. Every row must still go up, each
    # request within --chunk-rows, the parts sharing the run_id.
    sent: list[str] = []

    class Lake:
        def __init__(self, base, token):
            pass

        def upload(self, text, label):
            sent.append(text)
            return {"upload_id": "u", "row_count": 0}

    path = tmp_path / "results.csv"
    path.write_text(_csv(["run-a"] * 5), encoding="utf-8")
    monkeypatch.setattr(uploader, "check_paths", lambda _: _Report())
    monkeypatch.setattr(uploader, "format_report", lambda _: "checked")
    monkeypatch.setattr(uploader, "AuthoritativeLake", Lake)
    rc = uploader.main(["gpu", str(path), "--host", "https://example.invalid",
                        "--token", "hapat_ok", "--private", "--chunk-rows", "2"])

    assert rc == 0
    bodies = [text.splitlines()[1:] for text in sent]
    assert [len(b) for b in bodies] == [2, 2, 1]
    assert all("run-a" in line for b in bodies for line in b)
