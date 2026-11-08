from __future__ import annotations

import pytest

import tools.upload_to_authoritative_lake as uploader
from tests.corpus_row import row, to_csv


class _Report:
    def ok(self) -> bool:
        return True


def _csv(run_ids: list[str]) -> str:
    return to_csv([row(run_id=run_id) for run_id in run_ids])


def test_the_uploader_uses_the_public_review_flow(monkeypatch, tmp_path):
    calls: list[tuple[str, dict]] = []

    class Lake:
        def __init__(self, base, token):
            assert base == "https://example.invalid"
            assert token == "hapat_ok"

        def post(self, path, payload):
            calls.append((path, payload))
            if path == "/api/bench/upload":
                return {"upload_id": "22222222-2222-4222-8222-222222222222",
                        "row_count": 1}
            if path == "/api/bench/promote":
                return {"promoted": [{"batches": 1}]}
            if path.startswith("/api/bench/"):
                return {}

        def upload(self, text, label):
            return self.post("/api/bench/upload",
                             {"csv_text": text, "label": label})

        def share_within_peak(self, owner, upload_id):
            self.post(f"/api/bench/{upload_id}/contribute?within_peak=true", {})
            return self.post("/api/bench/promote", {"sets": [
                {"owner": owner, "kind": "upload", "id": upload_id}]})

    path = tmp_path / "results.csv"
    path.write_text(_csv(["run-a", "run-b"]), encoding="utf-8")
    monkeypatch.setattr(uploader, "check_paths", lambda _: _Report())
    monkeypatch.setattr(uploader, "format_report", lambda _: "checked")
    monkeypatch.setattr(uploader, "AuthoritativeLake", Lake)
    rc = uploader.main([
        "gpu", str(path), "--host", "https://example.invalid", "--token", "hapat_ok",
        "--owner", "11111111-1111-4111-8111-111111111111"])

    assert rc == 0
    assert [path for path, _ in calls] == [
        "/api/bench/upload",
        "/api/bench/22222222-2222-4222-8222-222222222222/contribute?within_peak=true",
        "/api/bench/promote",
        "/api/bench/upload",
        "/api/bench/22222222-2222-4222-8222-222222222222/contribute?within_peak=true",
        "/api/bench/promote",
    ]
    promote = [body for path, body in calls if path == "/api/bench/promote"]
    assert all(body["sets"][0]["owner"] == "11111111-1111-4111-8111-111111111111"
               for body in promote)


def test_the_local_service_is_not_accepted_as_authoritative(monkeypatch):
    monkeypatch.setattr(uploader, "_token", lambda _args: "hapat_ok")
    with pytest.raises(ValueError, match=r"not 127\.0\.0\.1:8100"):
        uploader.main(["gpu", "x.csv", "--host", "http://127.0.0.1:8100",
                       "--private"])
