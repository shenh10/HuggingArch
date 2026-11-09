import io
import urllib.error

import pytest

from backend import (gallery_cache_artifact, release_artifacts,
                     sol_pool_artifact, sweep_cache_artifact)
from backend.release_artifacts import artifact_to_pull, read_object

# What Supabase Storage answered for a missing object (probed 2026-11-04):
# HTTP 400 with a not-found body, not 404.
SUPABASE_MISSING = (
    400, b'{"statusCode":"404","error":"not_found",'
         b'"message":"Object not found","code":"NoSuchKey"}')


def _failing(code, body):
    def request(url, key, timeout):
        raise urllib.error.HTTPError(url, code, "x", {}, io.BytesIO(body))
    return request


@pytest.mark.parametrize("pinned", ["", "serving-law-v3/e/a.tar.gz"])
def test_a_missing_object_is_absent_not_an_error(pinned):
    assert artifact_to_pull(
        pinned, "serving-law-v3/e/index.json", "e", "https://s", "k",
        "sol-pool", 30, _failing(*SUPABASE_MISSING)) is None


def test_another_400_still_raises():
    with pytest.raises(urllib.error.HTTPError):
        read_object("https://s/x", "k", 30,
                    _failing(400, b'{"error":"InvalidJWT"}'))


@pytest.mark.parametrize("module", [sweep_cache_artifact, sol_pool_artifact,
                                    gallery_cache_artifact])
def test_startup_pull_without_an_index_succeeds(module, monkeypatch, capsys):
    # The 2026-11-04 crash loop: a new engine deployed before its index was
    # published made `pull` exit 1, and the entrypoint exits with it.
    monkeypatch.setenv("SUPABASE_URL", "https://s")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "k")
    monkeypatch.setenv("HA_SOL_POOL_ARTIFACT", "")
    monkeypatch.setenv("HA_GALLERY_CACHE_ARTIFACT", "")
    monkeypatch.setenv("HA_SWEEP_CACHE_ARTIFACT", "")
    monkeypatch.setattr(release_artifacts, "request",
                        _failing(*SUPABASE_MISSING))
    assert module.pull(timeout=1) == 0
    # Not an error, but said: this layer starts cold.
    assert "NONE PUBLISHED" in capsys.readouterr().err
