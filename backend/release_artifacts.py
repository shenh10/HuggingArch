"""Release bootstrap artifacts: one mechanism for every layer a new machine
pulls at container start instead of recomputing.

An artifact is an immutable, digest-addressed ``.tar.gz`` in Supabase
Storage, named by a small mutable index per engine version
(``<prefix>/<engine>/index.json``). Publishing writes the archive and its
``.sha256`` first and moves the index last; pulling reads the index,
downloads the archive, verifies its digest and installs it. A missing index
or pinned object means the layer falls back to runtime warm; a
present-but-invalid artifact fails startup (the entrypoint exits with it).

Two install shapes share this transport:

  * the SOL pool (`backend.sol_pool_artifact`) is a complete immutable
    corpus: the archive replaces the directory;
  * a result cache (`JsonCacheArtifact`: the Gallery's and the Scaling
    sweep's) is mergeable: the archive's entries are merged into the cache
    directory beside what runtime warm added, with the code traces those
    entries name (`backend.measure.cache`: an entry is served only while the
    running code satisfies its trace, and a fresh machine never recorded
    it).

Uploads go through ``urllib``'s default opener, so a proxy is used exactly
when the environment configures one, and directly otherwise. An object
of any size goes through Supabase's resumable (TUS) endpoint in fixed
chunks, each retried after re-reading the server's offset: a single POST of
a ~50 MB archive timed out on a slow uplink, and one path for every size is
one path to get right.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
from typing import Any, Callable, Iterable, Optional
import urllib.error
import urllib.request

from backend.code_trace.provider import TRACE_DIR_ENV, TRACES, Trace
from backend.measure.cache import TRACE_FIELD, DiskBackend


INDEX_SCHEMA = 1
DEFAULT_BUCKET = "sol-pool"
BUCKET_ENV = "HA_ARTIFACT_BUCKET"
#: Supabase's resumable-upload chunk: every PATCH but the last carries
#: exactly this many bytes.
TUS_CHUNK = 6 * 1024 * 1024


def bucket() -> str:
    return os.environ.get(BUCKET_ENV, DEFAULT_BUCKET)


# ---------------------------------------------------------------------------
# Reading.
# ---------------------------------------------------------------------------


def request(url: str, key: str, timeout: float):
    """An authorised GET of a storage object."""
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Accept": "application/octet-stream",
    })
    return urllib.request.urlopen(req, timeout=timeout)


def download(url: str, key: str, destination: Path, timeout: float) -> None:
    with request(url, key, timeout) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def valid_artifact_path(path: object) -> bool:
    return (
        isinstance(path, str)
        and bool(path)
        and not path.startswith("/")
        and ".." not in Path(path).parts
        and path.endswith(".tar.gz")
    )


def read_object(url: str, key: str, timeout: float,
                request: Callable[..., Any]) -> Optional[bytes]:
    """The object's bytes, or ``None`` when storage has no such object.

    Supabase Storage answers a missing object with HTTP 400 and a not-found
    body (``"code":"NoSuchKey"``), not with 404; both mean absent. Any other
    failure raises.
    """
    try:
        with request(url, key, timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        if exc.code == 400:
            body = exc.read().decode(errors="replace")
            if "NoSuchKey" in body or "Object not found" in body:
                return None
        raise


def artifact_to_pull(configured: str, index_path: str, expected_engine: str,
                     base: str, key: str, bucket: str, timeout: float,
                     request: Callable[..., Any]
                     ) -> Optional[tuple[str, str]]:
    """``(path, sha256)`` of the archive to install, ``None`` when absent.

    ``configured`` pins an archive path (its digest is the ``.sha256``
    sibling); empty reads this engine version's index.
    """
    if not configured:
        return selected_artifact(index_path, expected_engine, base, key,
                                 bucket, timeout, request)
    digest = read_object(f"{base}/storage/v1/object/{bucket}/{configured}.sha256",
                         key, timeout, request)
    if digest is None:
        return None
    return configured, digest.decode().strip().split()[0]


def selected_artifact(index_path: str, expected_engine: str, base: str,
                      key: str, bucket: str, timeout: float,
                      request: Callable[..., Any]
                      ) -> Optional[tuple[str, str]]:
    raw = read_object(f"{base}/storage/v1/object/{bucket}/{index_path}", key,
                      timeout, request)
    if raw is None:
        return None
    index = json.loads(raw)
    if index.get("schema") != INDEX_SCHEMA:
        raise ValueError("release artifact index has an unknown schema")
    if index.get("engine_version") != expected_engine:
        raise ValueError("release artifact index names another engine version")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("release artifact index has no artifacts")
    selected = artifacts[0]
    path, digest = selected.get("path"), selected.get("sha256")
    if not valid_artifact_path(path) or not isinstance(digest, str):
        raise ValueError("release artifact index has an invalid artifact")
    return path, digest


def fetch_verified(label: str, prefix: str, pin_env: str, destination: Path,
                   installed: Callable[[str, str], bool],
                   install: Callable[[Path, str, str], None],
                   timeout: float) -> int:
    """The one pull: select this engine's archive (``pin_env`` pins one),
    skip when ``installed(artifact, sha)``, else download beside
    ``destination``, verify its digest and ``install(archive, artifact,
    sha)``. 0 on success or nothing to pull; a bad artifact raises."""
    from backend.fingerprint import cache_engine_version
    from backend.supabase_env import configured, service_role
    if not configured():
        print(f"{label} artifact: Supabase is not configured",
              file=sys.stderr, flush=True)
        return 0
    base, key = service_role()
    base = base.rstrip("/")
    engine = cache_engine_version()
    selected = artifact_to_pull(
        os.environ.get(pin_env, ""), f"{prefix}/{engine}/index.json", engine,
        base, key, bucket(), min(30.0, timeout), request)
    if selected is None:
        # Not an error (a new engine deploys before its artifact is
        # published; failing would crash-loop the machine), but never quiet:
        # every view this layer serves will be computed cold.
        print(f"{label} artifact: NONE PUBLISHED for engine {engine} "
              f"({prefix}/{engine}/index.json); this layer starts cold",
              file=sys.stderr, flush=True)
        return 0
    artifact, expected = selected
    if installed(artifact, expected):
        print(f"{label} artifact: {destination} already has {artifact}",
              flush=True)
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / f".{destination.name}.{os.getpid()}.tar.gz"
    try:
        print(f"{label} artifact: downloading {artifact}", flush=True)
        download(f"{base}/storage/v1/object/{bucket()}/{artifact}", key,
                 archive, timeout)
        actual = digest(archive)
        if actual != expected:
            raise ValueError(
                f"{label} artifact digest mismatch: {actual} != {expected}")
        install(archive, artifact, expected)
    finally:
        archive.unlink(missing_ok=True)
    return 0


# ---------------------------------------------------------------------------
# Writing.
# ---------------------------------------------------------------------------


def _tar_info(name: str, size: int, *, directory: bool = False):
    info = tarfile.TarInfo(name)
    info.size = 0 if directory else size
    info.mode = 0o755 if directory else 0o644
    info.mtime = 0
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    return info


def write_archive(destination: Path, root: str,
                  members: Iterable[tuple[str, bytes]]) -> None:
    """A deterministic ``.tar.gz``: ``root/`` and each ``(relative, data)``
    below it in the order given, every parent directory once before its
    first file, zero mtimes — one content, one digest."""
    with destination.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0) as zipped, tarfile.open(
            fileobj=zipped, mode="w") as tar:
        tar.addfile(_tar_info(root, 0, directory=True))
        directories = {root}
        for relative, data in members:
            parts = relative.split("/")
            for depth in range(1, len(parts)):
                name = root + "/" + "/".join(parts[:depth])
                if name not in directories:
                    directories.add(name)
                    tar.addfile(_tar_info(name, 0, directory=True))
            tar.addfile(_tar_info(f"{root}/{relative}", len(data)),
                        io.BytesIO(data))


def _open(req: urllib.request.Request, timeout: float):
    # The default opener: the environment's proxy when one is configured,
    # a direct connection otherwise.
    return urllib.request.urlopen(req, timeout=timeout)


def upload(base: str, key: str, bucket: str, path: str, data: bytes,
           content_type: str, *, chunk: Optional[int] = None,
           retries: int = 30, wait: float = 5.0) -> None:
    """Store ``data`` at ``bucket/path`` (upsert) by a resumable (TUS)
    upload, whatever its size: ``chunk`` (`TUS_CHUNK`) bytes per PATCH; a
    failed PATCH is resent from the offset the server reports, up to
    ``retries`` failures in a row."""
    chunk = TUS_CHUNK if chunk is None else chunk
    tus = {"Authorization": f"Bearer {key}", "apikey": key,
           "x-upsert": "true", "Tus-Resumable": "1.0.0"}

    def b64(text: str) -> str:
        return base64.b64encode(text.encode()).decode()

    metadata = ",".join(f"{name} {b64(value)}" for name, value in {
        "bucketName": bucket, "objectName": path,
        "contentType": content_type, "cacheControl": "3600"}.items())
    create = urllib.request.Request(
        f"{base}/storage/v1/upload/resumable", data=b"", method="POST",
        headers={**tus, "Upload-Length": str(len(data)),
                 "Upload-Metadata": metadata, "Content-Length": "0"})
    with _open(create, 120) as response:
        location = response.headers["Location"]
    if location.startswith("/"):
        location = base + location
    offset, failures = 0, 0
    while offset < len(data):
        patch = urllib.request.Request(
            location, data=data[offset:offset + chunk], method="PATCH",
            headers={**tus, "Upload-Offset": str(offset),
                     "Content-Type": "application/offset+octet-stream"})
        try:
            with _open(patch, 600) as response:
                offset = int(response.headers["Upload-Offset"])
            failures = 0
            print(f"uploaded {offset}/{len(data)} bytes of {path}",
                  file=sys.stderr, flush=True)
        except (OSError, urllib.error.URLError) as error:
            failures += 1
            if failures > retries:
                raise
            print(f"upload chunk failed ({error!r:.200}); retry {failures}",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
            head = urllib.request.Request(location, method="HEAD",
                                          headers=tus)
            try:
                with _open(head, 120) as response:
                    offset = int(response.headers["Upload-Offset"])
            except (OSError, urllib.error.URLError):
                # Counted like a failed PATCH; the next PATCH is sent at the
                # last confirmed offset, which the server refuses if stale.
                failures += 1
                if failures > retries:
                    raise
    if offset != len(data):
        raise RuntimeError(
            f"upload of {path} ended at {offset} of {len(data)} bytes")


def publish(prefix: str, build: Callable[[Path, Path], int], root: Path,
            count_field: str) -> str:
    """Build ``root``'s archive, upload it, its ``.sha256`` and then this
    engine version's index naming it; the archive's storage path."""
    from backend.fingerprint import cache_engine_version
    from backend.supabase_env import configured, service_role
    if not configured():
        raise RuntimeError("Supabase is not configured")
    base, key = service_role()
    base = base.rstrip("/")
    engine = cache_engine_version()
    target = bucket()
    with tempfile.TemporaryDirectory(prefix="release-artifact.") as temporary:
        archive = Path(temporary) / "artifact.tar.gz"
        count = build(root, archive)
        sha = digest(archive)
        path = f"{prefix}/{engine}/{sha}.tar.gz"
        print(f"{path}: {archive.stat().st_size} bytes, {count} "
              f"{count_field}", file=sys.stderr, flush=True)
        # The archive before what names it: a reader that finds the digest
        # or the index finds the archive too.
        upload(base, key, target, path, archive.read_bytes(),
               "application/octet-stream")
        upload(base, key, target, f"{path}.sha256", f"{sha}\n".encode(),
               "text/plain")
        index = {"schema": INDEX_SCHEMA, "engine_version": engine,
                 "artifacts": [{"path": path, "sha256": sha,
                                count_field: count}]}
        upload(base, key, target, f"{prefix}/{engine}/index.json",
               json.dumps(index, sort_keys=True).encode() + b"\n",
               "application/json")
        return path


# ---------------------------------------------------------------------------
# A mergeable result cache.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheTier:
    """One `DiskBackend` directory of a result cache: where it sits under
    the cache directory (``""`` for the directory itself) and whether its
    files are compressed. The file format is the backend's; this names the
    backend, so the runtime cache and the artifact build the same one."""

    path: str = ""
    compress: bool = False

    def disk(self, root: Path, **bounds: Any) -> DiskBackend:
        return DiskBackend(root / self.path, compress=self.compress, **bounds)


@dataclass(frozen=True)
class JsonCacheArtifact:
    """A result-cache directory (`backend.measure.cache.DiskBackend` files)
    published and pulled as one artifact.

    ``entries`` are the tiers (`CacheTier`, relative to the directory) of
    the shareable entries; anything else under it is machine-local and
    never ships. ``machine_local`` names tiers whose entries a newly
    installed artifact clears (they were derived from what it replaces).
    The directory and its tiers are declared here once: the runtime cache
    builds its disk backends with `disk()`, and the artifact reads each
    tier's files and values through the same backend — an archive member
    is a tier's file, byte for byte.
    """

    name: str
    prefix: str
    dir_env: str
    default_dir: str
    pin_env: str
    entries: tuple[CacheTier, ...] = (CacheTier(),)
    machine_local: tuple[CacheTier, ...] = ()

    @property
    def volume_dirs(self) -> tuple[str, ...]:
        """The directories (by environment variable) a pull writes into:
        the entries' and the trace bodies they need. Both must outlive a
        restart, or the next start serves nothing it pulled."""
        return (self.dir_env, TRACE_DIR_ENV)

    def directory(self) -> Path:
        return Path(os.environ.get(
            self.dir_env,
            str(Path.home() / ".cache" / "huggingarch" / self.default_dir)))

    def disk(self, path: str = "", **bounds: Any) -> DiskBackend:
        """The runtime disk backend of the tier at ``path``, under
        `directory()`, with its bounds (``max_files``, ``max_bytes``)."""
        for tier in (*self.entries, *self.machine_local):
            if tier.path == path:
                return tier.disk(self.directory(), **bounds)
        raise KeyError(f"{self.name} cache declares no tier {path!r}")

    def _tier_files(self, root: Path) -> list[tuple[CacheTier, Path]]:
        """Each shareable entry under ``root`` with its tier, in archive
        order."""
        return sorted(
            ((tier, path) for tier in self.entries
             for path in tier.disk(root).files()),
            key=lambda item: item[1].relative_to(root).as_posix())

    def files(self, root: Path) -> list[Path]:
        """The shareable entries under ``root``: each tier's entry files
        (`DiskBackend.files`)."""
        return [path for _, path in self._tier_files(root)]

    # -- publishing -------------------------------------------------------

    def build(self, root: Path, destination: Path) -> int:
        """The archive of ``root``'s entries and, under ``code-traces/``,
        each trace they name, once; the entry count."""
        files = self._tier_files(root)
        if not files:
            raise ValueError(f"no shareable {self.name} entries under {root}")
        traces: set[str] = set()

        def members():
            for tier, path in files:
                data = path.read_bytes()
                value = tier.disk(root).decode(data)
                tid = (value.get(TRACE_FIELD) if isinstance(value, dict)
                       else None)
                if tid:
                    traces.add(tid)
                yield path.relative_to(root).as_posix(), data
            for tid in sorted(traces):
                trace = TRACES.get(tid)
                if trace is None:
                    raise ValueError(
                        f"{self.name} entries name trace {tid}, which this "
                        "machine does not keep: a pulling machine could not "
                        "serve them")
                yield (f"code-traces/{tid}.json",
                       json.dumps(trace.to_json()).encode())

        write_archive(destination, self.name, members())
        return len(files)

    def publish(self, root: Path) -> str:
        return publish(self.prefix, self.build, root, "files")

    # -- pulling ----------------------------------------------------------

    def check(self, archive: Path) -> int:
        """Accept only regular files below ``<name>/``: a declared tier's
        entry files (its directory, its backend's suffix) and the trace
        bodies under ``code-traces/``; the entry count."""
        files = 0
        suffixes = {tier.path: tier.disk(Path()).suffix
                    for tier in self.entries}
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                name = member.name.rstrip("/")
                if name != self.name and not name.startswith(f"{self.name}/"):
                    raise ValueError(
                        f"{self.name} cache archive member {name!r}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"{self.name} cache archive member is "
                                     f"not a file: {name}")
                relative = name[len(self.name) + 1:]
                parent, _, leaf = relative.rpartition("/")
                if parent == "code-traces":
                    if not leaf.endswith(".json"):
                        raise ValueError(f"{self.name} cache archive trace "
                                         f"is not JSON: {name}")
                    continue
                suffix = suffixes.get(parent)
                if suffix is None or not leaf.endswith(suffix):
                    raise ValueError(f"{self.name} cache archive member is "
                                     f"not an entry of a declared tier: "
                                     f"{name}")
                files += 1
            if not any(m.name.rstrip("/") == self.name for m in members):
                raise ValueError(
                    f"{self.name} cache archive has no {self.name}/ root")
            if not files:
                raise ValueError(
                    f"{self.name} cache archive has no entries")
        return files

    def installed(self, root: Path, artifact: str, sha: str) -> bool:
        """Whether ``root`` already holds this artifact: its marker, at
        least its entry count (runtime warm adds more; fewer means the
        cache evicted some, so merge again) and every trace it shipped."""
        try:
            marker = json.loads(
                (root / ".bootstrap.json").read_text(encoding="utf-8"))
            expected = int(marker["files"])
            traces = list(marker["traces"])
        except (OSError, ValueError, TypeError, KeyError,
                json.JSONDecodeError):
            return False
        if marker.get("artifact") != artifact or marker.get("sha256") != sha:
            return False
        return (expected > 0 and len(self.files(root)) >= expected
                and all(TRACES.get(tid) is not None for tid in traces))

    def install(self, archive: Path, root: Path, artifact: str,
                sha: str) -> int:
        """Merge the archive's entries into ``root`` and its traces into
        `TRACES`; the entry count."""
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{self.name}-cache.",
                                        dir=root.parent))
        try:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(staging, filter="data")
            unpacked = staging / self.name
            files = self.files(unpacked)
            if not files:
                raise ValueError(
                    f"{self.name} cache archive has no entries")
            for tier in self.machine_local:
                tier.disk(root).clear()
            for source in files:
                destination = root / source.relative_to(unpacked)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(
                    f".{destination.name}.{os.getpid()}.tmp")
                try:
                    # A fresh mtime: the disk tier evicts oldest first, and
                    # an entry installed now is as new as one computed now.
                    shutil.copyfile(source, temporary)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            traces = []
            for body in sorted((unpacked / "code-traces").glob("*.json")):
                trace = Trace.from_json(json.loads(
                    body.read_text(encoding="utf-8")))
                if trace.id != body.stem:
                    raise ValueError(f"{self.name} cache archive trace "
                                     f"{body.stem} has id {trace.id}")
                TRACES.put(trace)
                traces.append(trace.id)
            marker = root / ".bootstrap.json"
            temporary = root / f".bootstrap.json.{os.getpid()}.tmp"
            try:
                temporary.write_text(json.dumps(
                    {"artifact": artifact, "sha256": sha,
                     "files": len(files), "traces": traces},
                    sort_keys=True) + "\n", encoding="utf-8")
                os.replace(temporary, marker)
            finally:
                temporary.unlink(missing_ok=True)
            return len(files)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def pull(self, timeout: float = 900.0) -> int:
        root = self.directory()

        def install(archive: Path, artifact: str, sha: str) -> None:
            files = self.check(archive)
            merged = self.install(archive, root, artifact, sha)
            print(f"{self.name} cache artifact: merged {merged}/{files} "
                  f"files into {root}", flush=True)

        return fetch_verified(
            f"{self.name} cache", self.prefix, self.pin_env, root,
            lambda artifact, sha: self.installed(root, artifact, sha),
            install, timeout)

    def main(self, argv: Optional[list[str]] = None) -> int:
        parser = argparse.ArgumentParser(
            description=f"The {self.name} result cache's release artifact.")
        parser.add_argument("command", choices=["pull", "publish"])
        parser.add_argument("--timeout", type=float, default=900.0)
        parser.add_argument("--cache-dir", type=Path,
                            help="publish: the cache directory to publish")
        arguments = parser.parse_args(argv)
        if arguments.command == "pull":
            return self.pull(arguments.timeout)
        if arguments.cache_dir is None:
            parser.error("publish needs --cache-dir")
        print(self.publish(arguments.cache_dir.resolve()), flush=True)
        return 0
