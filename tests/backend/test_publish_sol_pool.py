import tarfile
import json

from scripts.publish_sol_pool import build
from backend.sol_pool_artifact import _expand_traces


def test_sol_artifact_is_digestable_and_keeps_bundle_files(tmp_path):
    root = tmp_path / "sol"
    bundle = root / "ab" / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    (bundle / "records-00000.jsonl").write_text("{}\n", encoding="utf-8")
    archive = tmp_path / "pool.tar.gz"

    assert build(root, archive) == 1
    with tarfile.open(archive, "r:gz") as tar:
        assert tar.getnames() == [
            "sol", "sol/ab", "sol/ab/bundle",
            "sol/ab/bundle/manifest.json",
            "sol/ab/bundle/records-00000.jsonl",
        ]


def test_trace_bodies_ship_factored_and_expand_back(tmp_path):
    root = tmp_path / "sol"
    first = root / "ab" / "first"
    second = root / "cd" / "second"
    for bundle in (first, second):
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    # The same function inventory appears in both bundles' traces: the
    # archive must carry each pair once and both traces as rows of it.
    (first / "traces.json").write_text(json.dumps({
        "trace1": [["a.py::f", "h1"],
                   ["a.py::<binding>:G", "h2"]]}), encoding="utf-8")
    (second / "traces.json").write_text(json.dumps({
        "trace2": [["a.py::<binding>:G", "h2"]]}), encoding="utf-8")
    archive = tmp_path / "pool.tar.gz"

    assert build(root, archive) == 2
    with tarfile.open(archive, "r:gz") as tar:
        assert tar.getnames() == [
            "sol", "sol/traces-table.json", "sol/trace-blocks.json",
            "sol/ab", "sol/ab/first", "sol/ab/first/manifest.json",
            "sol/ab/first/traces.refs.json",
            "sol/cd", "sol/cd/second", "sol/cd/second/manifest.json",
            "sol/cd/second/traces.refs.json",
        ]
        table = json.loads(tar.extractfile("sol/traces-table.json").read())
        blocks = json.loads(tar.extractfile("sol/trace-blocks.json").read())
        first_refs = json.loads(
            tar.extractfile("sol/ab/first/traces.refs.json").read())
        second_refs = json.loads(
            tar.extractfile("sol/cd/second/traces.refs.json").read())
    assert table == {"schema": 1, "entries": [
        ["a.py::<binding>:G", "h2"], ["a.py::f", "h1"]]}
    # The binding set both traces share ships once, as a block of rows.
    (block,) = blocks["blocks"].values()
    assert block == [0]
    assert first_refs["trace1"]["rows"] == [1]
    assert second_refs["trace2"]["rows"] == []
    assert (first_refs["trace1"]["blocks"]
            == second_refs["trace2"]["blocks"]
            == list(blocks["blocks"]))

    # Install expands the rows back into each bundle's own format and
    # leaves the machine-local pool exactly as a warm run would have it.
    unpacked = tmp_path / "unpacked" / "sol"
    unpacked.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(unpacked.parent, filter="data")
    _expand_traces(unpacked)
    assert json.loads((unpacked / "ab/first/traces.json").read_text()) == {
        "trace1": [["a.py::<binding>:G", "h2"], ["a.py::f", "h1"]]}
    assert json.loads((unpacked / "cd/second/traces.json").read_text()) == {
        "trace2": [["a.py::<binding>:G", "h2"]]}
    assert not (unpacked / "traces-table.json").exists()
    assert not (unpacked / "trace-blocks.json").exists()
    assert not list(unpacked.glob("*/*/traces.refs.json"))
