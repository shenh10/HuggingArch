"""Box profiles — the machine-local facts a measurement run is given.

Not marked `regression`: these are the ratchet on a box file that is only ever
exercised on the box itself. A wrong relative path or a fact that leaked from
one env into all of them costs a GPU night to discover there and a second to
discover what it did to the rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernel_bench.box import (BoxProfileError, apply_library_env,
                              apply_profile_env, find_box_profile,
                              library_requirements, load_box_profile)

_BOXES = Path(__file__).resolve().parents[2] / "kernel_bench" / "envs" / "boxes"
#: The box whose kernel driver (CUDA 12.9) is older than the CUDA its newest
#: provider envs are built against.
_H100 = _BOXES / "h100-sxm-cu128.toml"


class TestOneEnvsEnvironmentIsNotEveryEnvs:
    """`[env]` is what every env on the box runs in; `[library_env.<lib>]` is
    what ONE env runs in. The distinction exists because the forward-compat
    user-mode driver a CUDA-13 env needs would, box-wide, also become the
    driver every CUDA-12 env is measured against — a change to every future
    row of theirs that no row records."""

    def test_the_overlay_reaches_one_library_only(self, tmp_path):
        """Deliberately NOT one of the three libraries the H100 profile gives
        the compat driver to: an overlay that only works for the libraries
        someone had in mind is the special case this table exists not to be,
        and a test written on `sglang` would pass `if lib in (...)` too."""
        path = tmp_path / "box.toml"
        path.write_text(
            'name = "two-stacks"\n'
            '[env]\n'
            'CUDA_HOME = "/usr/local/cuda"\n'
            '[library_env.flashinfer]\n'
            'LD_LIBRARY_PATH = "${HOME}/compat"\n')
        profile = load_box_profile(path)
        base = apply_profile_env({"HOME": "/root"}, profile)

        assert apply_library_env(base, profile, "flashinfer")["LD_LIBRARY_PATH"] \
            == "/root/compat"
        # Including the libraries that DO carry it on the real box: what the
        # overlay reaches is what the profile says, not what the code knows.
        for lib in ("fa2", "sglang", "vllm", "marlin", "mamba_ssm"):
            assert "LD_LIBRARY_PATH" not in apply_library_env(base, profile, lib), lib
        # And the box-wide half still reaches both.
        for lib in ("flashinfer", "fa2"):
            assert apply_library_env(base, profile, lib)["CUDA_HOME"] \
                == "/usr/local/cuda"

    def test_a_profile_without_the_table_changes_nothing(self, tmp_path):
        path = tmp_path / "box.toml"
        path.write_text('name = "plain"\n')
        profile = load_box_profile(path)
        assert apply_library_env({"PATH": "/usr/bin"}, profile, "vllm") == \
            {"PATH": "/usr/bin"}
        assert apply_library_env({"PATH": "/usr/bin"}, None, "vllm") == \
            {"PATH": "/usr/bin"}

    def test_a_flat_value_is_refused_not_guessed(self, tmp_path):
        path = tmp_path / "box.toml"
        path.write_text('name = "bad"\n[library_env]\nsglang = "/compat"\n')
        with pytest.raises(BoxProfileError, match=r"library_env\.sglang"):
            load_box_profile(path)

    def test_an_empty_value_is_refused_rather_than_read_as_absence(self, tmp_path):
        """Written down, an empty string looks deliberate and behaves like the
        line not being there — `env_requirements` with `vllm = ""` resolves the
        generated env under the box constraints, which is what naming an
        override was meant to stop. Refused in both tables."""
        path = tmp_path / "box.toml"
        path.write_text('name = "bad"\n[env_requirements]\nvllm = ""\n')
        with pytest.raises(BoxProfileError, match="states nothing"):
            load_box_profile(path)
        path.write_text('name = "bad"\n'
                        '[library_env.vllm]\nLD_LIBRARY_PATH = "  "\n')
        with pytest.raises(BoxProfileError, match="states nothing"):
            load_box_profile(path)


class TestAnOverrideIsThatEnvsRequirementsInFull:
    """The box-wide `constraints` is the torch build for the envs that do not
    state one. An `[env_requirements]` override states one in full, so the
    default steps aside — both at once is unresolvable the moment the box has
    a second wheel stack (`torch==2.8.0+cu128` against vLLM's `torch==2.13.0`).
    """

    def _profile(self, tmp_path):
        path = tmp_path / "box.toml"
        path.write_text(
            'name = "two-stacks"\n'
            'constraints = "boxes/cu128.txt"\n'
            '[env_requirements]\n'
            'vllm = "boxes/box.vllm.txt"\n')
        return load_box_profile(path)

    def test_an_overridden_env_does_not_also_take_the_box_pin(self, tmp_path):
        profile = self._profile(tmp_path)
        env_file, constraints = library_requirements(profile, "vllm", "envs")
        assert env_file == "boxes/box.vllm.txt"
        assert constraints is None

    def test_an_env_the_box_says_nothing_about_still_takes_it(self, tmp_path):
        profile = self._profile(tmp_path)
        env_file, constraints = library_requirements(profile, "fla", "envs")
        assert env_file == str(Path("envs") / "fla.txt")
        assert constraints == "boxes/cu128.txt"

    def test_a_constraints_file_the_operator_PASSED_reaches_every_env(self, tmp_path):
        """The profile's `constraints` is the box's default and steps aside for
        an override. `--constraints` is not a default: it is the reach for when
        a driver is misbehaving, and an env that cannot satisfy it must fail to
        resolve rather than quietly resolve without it."""
        profile = self._profile(tmp_path)
        for lib in ("vllm", "fla"):
            _, constraints = library_requirements(
                profile, lib, "envs", asked_constraints="/tmp/driver-safe.txt")
            assert constraints == "/tmp/driver-safe.txt", lib
        # …and it does not change which env file is used.
        assert library_requirements(
            profile, "vllm", "envs",
            asked_constraints="/tmp/driver-safe.txt")[0] == "boxes/box.vllm.txt"

    def test_no_profile_falls_back_to_the_generated_env(self):
        assert library_requirements(None, "fla", "envs") == (
            str(Path("envs") / "fla.txt"), None)


class TestTheH100BoxSaysWhatItMeans:
    """Ratchets on the real file. Everything here was a live defect once."""

    def test_every_referenced_requirements_file_exists(self):
        """Including the ones a box file pulls in with `-r`: a box override
        that says `-r ../vllm.txt` is only resolved by uv, on the box, at the
        start of a GPU night."""
        profile = load_box_profile(_H100)
        root = Path(_H100).resolve().parents[3]
        for lib, rel in profile.env_requirements.items():
            path = root / rel
            assert path.is_file(), f"{lib}: {rel} is not in the repository"
            for line in path.read_text().splitlines():
                if line.startswith("-r "):
                    included = (path.parent / line[3:].strip()).resolve()
                    assert included.is_file(), \
                        f"{lib}: {rel} includes {line[3:].strip()}, which is not there"
        assert (root / profile.constraints).is_file()

    def test_the_cuda13_envs_get_the_compat_driver_and_the_others_do_not(self):
        """The three envs whose wheels are CUDA-13 (vLLM's extensions, and
        sglang's from 0.5.11 via `cuda-python>=13`) name the forward-compat
        user-mode driver; the CUDA-12 envs must not, or their rows stop being
        comparable with the ones already measured."""
        profile = load_box_profile(_H100)
        base = apply_profile_env({"KERNEL_BENCH_WORK_DIR": "/disk"}, profile)
        for lib in ("vllm", "marlin", "sglang"):
            env = apply_library_env(base, profile, lib)
            assert env.get("LD_LIBRARY_PATH") == "/disk/cuda-compat-13.0/compat", lib
        for lib in ("fa2", "fla", "flashinfer", "humming"):
            assert "LD_LIBRARY_PATH" not in apply_library_env(base, profile, lib), lib
        assert "LD_LIBRARY_PATH" not in base, \
            "box-wide would re-time every cu128 env against a driver no row names"

    def test_the_cuda13_envs_resolve_off_pypi_not_the_cu128_index(self):
        """The pytorch index publishes local-version companions
        (`torchaudio-2.11.0+cu128`), and a local version sorts above the plain
        one — so with the box's extra index in scope, `unsafe-best-match`
        satisfied sglang's `torchaudio==2.11.0` from cu128 next to a CUDA-13
        torch. The env installed and then died on first import."""
        profile = load_box_profile(_H100)
        base = apply_profile_env({"KERNEL_BENCH_WORK_DIR": "/disk"}, profile)
        assert "cu128" in base["UV_EXTRA_INDEX_URL"]
        for lib in ("vllm", "marlin", "sglang"):
            env = apply_library_env(base, profile, lib)
            # The URL itself, not merely "not cu128": the profile says an empty
            # index URL is not a declaration, so an empty or arbitrary value
            # has to be red here too.
            assert env["UV_EXTRA_INDEX_URL"] == "https://pypi.org/simple", lib
        assert "cu128" in apply_library_env(base, profile, "fla")["UV_EXTRA_INDEX_URL"]

    def test_the_cuda12_pin_is_not_layered_on_a_cuda13_env(self):
        """The regression this whole arrangement exists to prevent: with the
        box pin applied, `uv` cannot resolve the CUDA-13 envs at all."""
        profile = load_box_profile(_H100)
        for lib in ("vllm", "marlin", "sglang"):
            _, constraints = library_requirements(profile, lib, "kernel_bench/envs")
            assert constraints is None, lib
        _, constraints = library_requirements(profile, "fla", "kernel_bench/envs")
        assert constraints == profile.constraints


class TestTheChildProcessIsTheOneThatGetsIt:
    """The overlay is only worth anything if it reaches the process that
    measures. Both of `run-all`'s children — the pre-resolve and the supervised
    measurement — have to be launched with the library's environment, and
    nothing above would go red if either were handed the box-wide one instead.
    """

    def _run_all(self, tmp_path, monkeypatch, lib_env_lines):
        """`run-all` over a one-library manifest and a one-job plan, capturing the
        environment of every child it launches."""
        from kernel_bench import cli

        profile = tmp_path / "box.toml"
        profile.write_text(
            'name = "two-stacks"\n'
            f'work_dir = "{tmp_path}"\n'
            '[env]\n'
            'SHARED = "box-wide"\n'
            + lib_env_lines)
        # One job, planned for the one env (`plan.write_plan`'s annotation):
        # run-all starts only the envs a plan's candidates need.
        from kernel_bench.box import load_box_profile
        from tests.kernel_bench.test_plan_candidates import judged_on
        manifest = {"flashinfer": {"source_build": False, "build_hint": "",
                                   "import_names": ("flashinfer",),
                                   "providers": ("x",),
                                   "no_build_isolation": None}}
        (tmp_path / "jobs.jsonl").write_text(json.dumps({
            "candidates": {"x": "flashinfer"},
            "judged_on": judged_on(load_box_profile(profile),
                                   manifest=manifest)}) + "\n")
        seen = []

        from kernel_bench.supervisor import SupervisedResult

        def _fake_run(command, *, output_path, env, cwd, budget_s,
                      stall_timeout_s, **kw):
            seen.append((list(command), dict(env)))
            return SupervisedResult(returncode=0, stopped_by=None, elapsed_s=0.0)

        monkeypatch.setattr("kernel_bench.supervisor.run_command", _fake_run)
        monkeypatch.setattr("kernel_bench.providers.dependency_manifest",
                            lambda: manifest)
        cli.main(["run-all", "--jobs", str(tmp_path / "jobs.jsonl"),
                  "-o", str(tmp_path / "out"), "--box", str(profile)])
        return seen

    def test_every_child_is_launched_with_the_librarys_environment(
            self, tmp_path, monkeypatch):
        seen = self._run_all(
            tmp_path, monkeypatch,
            '[library_env.flashinfer]\nLD_LIBRARY_PATH = "/disk/compat"\n')
        assert seen, "run-all launched nothing"
        for command, env in seen:
            assert env.get("LD_LIBRARY_PATH") == "/disk/compat", command[:4]
            assert env.get("SHARED") == "box-wide", command[:4]

    @pytest.mark.parametrize("inherited", ["/parent/cuda/lib", None])
    def test_a_library_the_box_says_nothing_about_gets_no_overlay(
            self, tmp_path, monkeypatch, inherited):
        # No overlay means inherit the parent, including a CUDA image's
        # library path; it does not mean erase that path from every child.
        if inherited is None:
            monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        else:
            monkeypatch.setenv("LD_LIBRARY_PATH", inherited)
        seen = self._run_all(
            tmp_path, monkeypatch,
            '[library_env.sglang]\nLD_LIBRARY_PATH = "/disk/compat"\n')
        assert seen, "run-all launched nothing"
        for command, env in seen:
            if inherited is None:
                assert "LD_LIBRARY_PATH" not in env, command[:4]
            else:
                assert env.get("LD_LIBRARY_PATH") == inherited, command[:4]


class TestTheQueueFindsTheProfileFromAnyWorkingDirectory:
    """supervise.sh hands the queue a repo-root-relative profile path while
    the queue and every provider env it spawns run from the work directory.
    The 2026-10-30 outage: the relative path missed from that CWD, the
    fallback re-joined it onto `boxes/`, and eight models burned in a loop on
    a "box profile not found" whose path nobody had passed."""

    _ENVS = Path(__file__).resolve().parents[2] / "kernel_bench" / "envs"

    def test_a_repo_relative_path_resolves_from_the_work_directory(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        profile = find_box_profile(
            "kernel_bench/envs/boxes/h100-sxm-cu128.toml", str(self._ENVS))
        assert profile is not None and profile.name == "h100-sxm-cu128"

    def test_a_bare_profile_name_still_resolves(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        profile = find_box_profile("h100-sxm-cu128", str(self._ENVS))
        assert profile is not None and profile.name == "h100-sxm-cu128"

    def test_a_missing_profile_reports_every_path_it_tried(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(BoxProfileError) as raised:
            find_box_profile("kernel_bench/envs/boxes/no-such-box.toml",
                             str(self._ENVS))
        message = str(raised.value)
        assert "kernel_bench/envs/boxes/no-such-box.toml" in message
        # The doubled join that ate a GPU night must not be the only answer.
        assert "boxes/kernel_bench/envs/boxes/no-such-box.toml" not in message
        assert "tried" in message


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
