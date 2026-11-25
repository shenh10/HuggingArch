"""
HF Metadata — a checkpoint's safetensors header, read from the HuggingFace Hub
or a local repo, and the totals it implies.

No torch needed. Downloads index.json and every shard's header (HTTP range
requests). Handles mixed-precision models (FP8 + BF16).

Sub-modules:
  header.py         `SafetensorsHeader` — every tensor the file holds, by the
                    name the file gives it; `ReadTensors` accumulates them
                    off the shard headers (a tensor copied into a second
                    shard is the same tensor, counted once)
  _cache.py         on-disk JSON cache (model_id → fetched record)

This module owns: `HUB_UNREACHABLE` (what "the Hub could not be read"
means, for every Hub reader), the public `fetch_config` / `fetch_metadata`
entries (a model's two serving inputs, read from one place in one order), the hub /
local-repo header readers, the record a snapshot stores (`metadata_record`), totals
(`summarize`), and `format_header` for prompt display.
"""
import json
import logging
from typing import Any, Callable, Dict, Optional

from huggingface_hub import errors as hf_errors

from backend.custom_id import is_custom_id, model_path, strip_prefix

from ._cache import cache_load_raw, cache_store
from .header import ReadTensors, SafetensorsHeader
from backend.arch.quant_schemes import (
    PackedStorageMismatch,
    resolve_scheme,
    SCHEME_NONE,
    QuantScheme,
    is_quantization_metadata,
    packed_unpack_factor,
    scheme_named,
)
from backend.arch.dtypes import dtype_bytes as _dtype_bytes

logger = logging.getLogger(__name__)

# The Hub could not be read: offline mode, or a failure of the hub's own HTTP
# client (DNS, connect, timeout, proxy, a dropped stream). That client is
# httpx2, whose errors are neither `httpx.HTTPError` nor `OSError`; a status
# answer (`HfHubHTTPError`, incl. not-found / gated) subclasses
# `hf_errors.HTTPError` too, so a caller that tells a status apart catches it
# first. The one definition every Hub reader in `backend/` catches.
HUB_UNREACHABLE = (
    hf_errors.OfflineModeIsEnabled,     # HF_HUB_OFFLINE
    hf_errors.HTTPError,                # the hub's HTTP client, any failure
)


def summarize(header: SafetensorsHeader, *, scheme: QuantScheme,
              num_bits: int,
              unknown_method: Optional[str] = None) -> Dict[str, Any]:
    """The header with the totals it implies: parameters and bytes, split
    into quant-metadata sidecars and the rest.

    The split is the header's own: a tensor is a sidecar by the selected
    scheme's suffixes (its config's `also_stores` layouts included), and
    only those; everything else — weights and any buffer the file serialises —
    is the rest. Which of the rest is a buffer is the spec's declaration,
    joined by name in `validate`, not a fact the header carries.
    """
    total_params = 0
    quantization_metadata_params = 0
    model_storage_bytes = 0
    quantization_metadata_bytes = 0
    for group in header:
        name, dtype = group.template, group.dtype
        elements = group.elements * group.count
        # A dtype not in the table raises rather than being priced at BF16:
        # see `arch.dtypes` for what a silent default once cost.
        width = _dtype_bytes(dtype)
        if is_quantization_metadata(name, scheme):
            quantization_metadata_params += elements
            quantization_metadata_bytes += elements * width
            continue
        total_params += elements * packed_unpack_factor(
            name, dtype, num_bits=num_bits, scheme=scheme)
        model_storage_bytes += elements * width

    out = {
        "header": header,
        "total_params": total_params,
        "quantization_metadata_params": quantization_metadata_params,
        "checkpoint_params_total": total_params + quantization_metadata_params,
        # The one byte count that is MEASURED rather than reconstructed: the
        # bytes the file's tensors occupy, summed from each shard header's
        # `data_offsets`. Everything else here is `elements x
        # _DTYPE_BYTES[dtype]`, this module's opinion of a file size; the
        # spans are the only figure that can contradict both dtype tables.
        "index_total_size": header.total_size,
        # The repo commit the header was read at: a repo's `main` is not a
        # fixed checkpoint (DeepSeek-V3 went FP8 → BF16 under the same name).
        "index_revision": header.revision,
        "model_storage_bytes": model_storage_bytes,
        "quantization_metadata_bytes": quantization_metadata_bytes,
        "checkpoint_storage_bytes": model_storage_bytes + quantization_metadata_bytes,
        "quant_scheme": scheme.name,
        "num_bits": num_bits,
    }
    if unknown_method:
        # The model declared a `quant_method` we don't have a YAML for —
        # we fell back to universal-suffix matching. Surface the raw method
        # name so callers can warn the user (and skip "exact match"
        # comparisons that might be coincidentally agreeing on a
        # double-undercount, the way mxfp4 did before its scheme landed).
        out["quant_method_unknown"] = unknown_method
    return out


def metadata_record(meta: Dict[str, Any]) -> Dict[str, Any]:
    """What a snapshot (or the on-disk cache, or a custom model) stores of a
    fetch: the header verbatim, and the spec-generation facts the header
    cannot carry — the quant scheme (and bit width) the writer resolved from
    the config, and a `quant_method` no scheme is registered for. Totals are
    not stored; `summarize` derives them on read."""
    record: Dict[str, Any] = {"header": meta["header"].to_json(),
                              "quant_scheme": meta["quant_scheme"],
                              "num_bits": meta["num_bits"]}
    if meta.get("quant_method_unknown"):
        record["quant_method_unknown"] = meta["quant_method_unknown"]
    return record


def metadata_from_record(record: Dict[str, Any], *, source: str,
                         legacy_config: Optional[Callable[[], Optional[dict]]] = None
                         ) -> Dict[str, Any]:
    """`summarize` over a stored record, under the scheme its writer
    resolved. A record without the verbatim header predates it and is
    refused: its names were folded on write, and the fold is not
    reversible."""
    raw = record.get("header")
    if not isinstance(raw, dict) or "tensors" not in raw:
        raise ValueError(
            f"{source} stores no verbatim safetensors header (it predates "
            f"them: its tensor names were folded when it was written). "
            f"Refetch it: `python -m backend.hf.snapshot <model_id>`, or "
            f"re-register a custom model with --hf-model-path.")
    header = SafetensorsHeader.from_json(raw)
    scheme = scheme_named(record["quant_scheme"])
    try:
        return summarize(header, scheme=scheme, num_bits=int(record["num_bits"]),
                         unknown_method=record.get("quant_method_unknown"))
    except PackedStorageMismatch:
        # Older writers persisted the wrapper scheme before resolving its
        # delegated packing format. Recover only a target the wrapper declares,
        # from the config stored alongside this same header.
        if not scheme.delegates_to or legacy_config is None:
            raise
        resolved, bits, unknown = resolve_scheme(legacy_config())
        if resolved.name not in {target for _, _, target in scheme.delegates_to}:
            raise
        return summarize(header, scheme=resolved, num_bits=bits, unknown_method=unknown)


def scan_local_hf_repo(repo_dir, *, scheme: "QuantScheme", num_bits: int,
                       unknown_method: Optional[str] = None) -> Dict[str, Any]:
    """Scan a local HuggingFace-format repo and produce the same metadata
    dict that `_fetch_metadata_hf` returns for hub models.

    Handles:
      • `model.safetensors.index.json` (sharded) or single `model.safetensors`
      • safetensors binary header parsing (little-endian u64 length + JSON)

    Reads headers only. The checkpoint carries dtype / shape / nbytes and no
    quant method; which scheme classifies its tensors is the caller's to say
    (`scheme` / `num_bits`, `resolve_scheme(config)`). Spec generation resolves one from `config.quantization_config`
    when it registers the model; serving never parses that field
    (CLAUDE.md §2.4) — its scheme identity is `ArchSpec.quant_context`.
    """
    from pathlib import Path as _Path

    repo = _Path(repo_dir)
    if not repo.is_dir():
        raise FileNotFoundError(f"Not a directory: {repo_dir}")

    idx_path = repo / "model.safetensors.index.json"
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text())
        shards = sorted(set((idx.get("weight_map") or {}).values()))
    else:
        shards = sorted(p.name for p in repo.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(
                f"No safetensors under {repo_dir} "
                f"(looked for model.safetensors.index.json and *.safetensors)"
            )

    read = ReadTensors()
    for shard in shards:
        path = repo / shard
        if not path.is_file():
            raise FileNotFoundError(
                f"{repo_dir}: the index lists shard {shard}, which is missing")
        for name, info in _read_safetensors_header(path).items():
            start, end = info.get("data_offsets", (0, 0))
            read.add(name, info.get("dtype", ""), info.get("shape", []),
                     int(end) - int(start), shard=shard)
    header = read.header()
    return summarize(header, scheme=scheme, num_bits=num_bits,
                     unknown_method=unknown_method)


def _read_safetensors_header(path) -> Dict[str, Dict[str, Any]]:
    """Parse a safetensors file's JSON header.

    Safetensors format: first 8 bytes are little-endian u64 header length,
    then that many bytes of UTF-8 JSON. Each key is a tensor name (plus
    `__metadata__` for free-form comments), each value holds dtype +
    shape + data_offsets.
    """
    import struct
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"{path}: truncated safetensors header")
        header_len = struct.unpack("<Q", raw_len)[0]
        raw_hdr = f.read(header_len)
    hdr = json.loads(raw_hdr.decode("utf-8"))
    hdr.pop("__metadata__", None)
    return hdr


def fetch_config(model_id: str, overlay_dir: Optional[str] = None
                 ) -> Optional[Dict[str, Any]]:
    """`model_id`'s config.json, or None when it cannot be read.

    The same order as `fetch_metadata`, so a model's config and checkpoint
    always come from the same record: a custom model's registered dir; else
    the snapshot (shared dir, then the overlay `overlay_dir` / the env
    names); else the Hub. Callers that must not proceed without a config
    raise on None; the reason is `analyzer.explain_config_fetch_failure`'s
    to classify."""
    if is_custom_id(model_id):
        path = model_path(strip_prefix(model_id)) / "config.json"
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            logger.error(f"Custom model not registered: {model_id}")
            return None
    from backend.arch.snapshots import load_config
    snap_cfg = load_config(model_id, overlay_dir)
    if snap_cfg is not None:
        return snap_cfg
    try:
        from huggingface_hub import hf_hub_download
        # EntryNotFoundError (repo exists, no config.json) is NOT an OSError
        # in current huggingface_hub, and HFValidationError is a ValueError
        # rather than an OSError. Name both explicitly or a bad repo id
        # escapes as a 500 instead of reaching the classifier.
        from huggingface_hub.errors import EntryNotFoundError, HFValidationError
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, EntryNotFoundError, HFValidationError,
            json.JSONDecodeError, *HUB_UNREACHABLE) as e:
        logger.error(f"Failed to fetch config for {model_id}: {e}")
        return None


def fetch_metadata(model_id: str, overlay_dir: Optional[str] = None
                   ) -> Dict[str, Any]:
    """`model_id`'s checkpoint header and the totals it implies: a custom
    model's registered record; else the snapshot (shared dir, then the
    overlay `overlay_dir` / the env names) under the scheme its writer
    resolved; else the Hub under no scheme."""
    # Custom-model passthrough: the local header record, if any. Identity
    # lives on `custom_id`; this module does not import registration / CLI.
    # Hub and custom still share this entry so callers receive the same
    # `header` + totals.
    if is_custom_id(model_id):
        root = model_path(strip_prefix(model_id))
        record_path = root / "checkpoint_header.json"
        if record_path.is_file():
            try:
                record = json.loads(record_path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                # The file exists but is corrupt — don't silently degrade to
                # zero params (wrong sizing everywhere downstream). A
                # genuinely un-indexed custom model simply has no file.
                raise ValueError(
                    f"custom model {model_id} checkpoint_header.json is "
                    f"corrupt: {e}") from e
            return metadata_from_record(
                record, source=f"custom model {model_id}", legacy_config=lambda: fetch_config(model_id))
        if (root / "tensor_patterns.json").is_file():
            return metadata_from_record(
                {}, source=f"custom model {model_id}")
        return {}
    # Git-managed snapshot: lets built-in (incl. gated) models work offline.
    from backend.arch.snapshots import load_snapshot
    snap = load_snapshot(model_id, overlay_dir)
    meta = snap.get("metadata") if snap else None
    if meta is not None:
        # The snapshot's sidecars are classified under the scheme its writer
        # resolved from the config (recorded beside the header); serving does
        # not parse the config itself (CLAUDE.md §2.4).
        return metadata_from_record(meta, source=f"snapshot {model_id}",
                                    legacy_config=lambda: snap.get("config"))
    # No snapshot: serving holds no checkpoint scheme, so no tensor is a
    # sidecar (`SCHEME_NONE`); spec generation snapshots a model first.
    return _fetch_metadata_hf(model_id, scheme=SCHEME_NONE, num_bits=0)


def read_header_hf(model_id: str, revision: Optional[str] = None, *,
                   read_from: Optional[str] = None) -> SafetensorsHeader:
    """Every shard header of `model_id` at `revision` (default: `main`).

    `read_from` names a public repo whose shard files are byte-identical to
    a gated `model_id`'s — the same LFS sha256 per shard and the same index
    blob, checked here — so its headers ARE the gated repo's.
    """
    import time as _time
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.model_info(model_id, revision=revision, files_metadata=True)
    files = {s.rfilename: s for s in info.siblings}
    source, source_rev = model_id, info.sha
    if read_from:
        mirror = api.model_info(read_from, files_metadata=True)
        mirror_files = {s.rfilename: s for s in mirror.siblings}
        source, source_rev = read_from, mirror.sha

    if "model.safetensors.index.json" in files:
        if read_from and (files["model.safetensors.index.json"].blob_id
                          != mirror_files["model.safetensors.index.json"].blob_id):
            raise ValueError(f"{read_from}'s index is not {model_id}'s")
        idx_path = hf_hub_download(repo_id=source,
                                   filename="model.safetensors.index.json",
                                   revision=source_rev)
        with open(idx_path) as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map") or {}
        shards = sorted(set(weight_map.values()))
    else:
        weight_map = None
        shards = ["model.safetensors"]
    if read_from:
        for shard in shards:
            if files[shard].lfs.sha256 != mirror_files[shard].lfs.sha256:
                raise ValueError(f"{read_from}'s {shard} is not {model_id}'s")

    read = ReadTensors()
    for shard in shards:
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                file_meta = api.parse_safetensors_file_metadata(
                    source, shard, revision=source_rev)
                break
            except (OSError, ValueError, *HUB_UNREACHABLE) as e:
                last_err = e
                _time.sleep(0.4 * (attempt + 1))
        else:
            raise OSError(f"{model_id}: shard {shard} header unreadable: {last_err}")
        for name, t in file_meta.tensors.items():
            read.add(name, str(t.dtype), t.shape,
                     int(t.data_offsets[1] - t.data_offsets[0]), shard=shard)
    if weight_map is not None and set(weight_map) != read.names:
        raise ValueError(
            f"{model_id}: the index and the shard headers name different "
            f"tensors ({len(set(weight_map) ^ read.names)} differ)")
    return read.header(revision=info.sha, read_from=read_from)


def _fetch_metadata_hf(model_id: str, *,
                       scheme: "QuantScheme", num_bits: int,
                       unknown_method: Optional[str] = None) -> Dict[str, Any]:
    """The hub checkpoint's header (`read_header_hf`) with its totals.

    The scheme that classifies tensor names and unpacks packed weights is the
    caller's (see `scan_local_hf_repo`): the snapshot writer resolves one at
    spec-generation time; the serving path reads headers under `SCHEME_NONE`
    and takes its quant identity from the spec's `quant_context`.
    Returns {} when the hub cannot be read.
    """
    cached = cache_load_raw(model_id)
    if cached is not None:
        header = SafetensorsHeader.from_json(cached["header"])
    else:
        try:
            header = read_header_hf(model_id)
        except (OSError, ValueError, *HUB_UNREACHABLE) as e:
            logger.warning(
                f"Failed to fetch safetensors metadata for {model_id}: {e}")
            return {}
    result = summarize(header, scheme=scheme, num_bits=num_bits,
                       unknown_method=unknown_method)
    if cached is None:
        cache_store(model_id, metadata_record(result))
    return result


def format_header(header: Optional[SafetensorsHeader]) -> str:
    """The header as a prompt reads it: one line per group, its name with
    each numeric segment's index set written in, then shape, dtype and how
    many tensors it holds."""
    if not header:
        return ""
    return "\n".join(
        f"  {g.display()}: shape={list(g.shape)} dtype={g.dtype} ×{g.count}"
        for g in sorted(header, key=lambda g: (g.template, g.dtype, g.shape)))
