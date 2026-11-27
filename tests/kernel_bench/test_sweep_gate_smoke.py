"""The shape-sweep feasibility gate, unmocked.

`kernel_bench.sweep._ceilings_for_deployment` calls the capacity layer for
the deployment it gates. The sweep tests mock it, so a call that no longer type-checks would pass every
one of them while every real sweep died at the gate (kimi S4a/b review,
must-fix 1). This runs the real call once.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = PROJECT_ROOT / "backend" / "arch" / "snapshots" / "Qwen__Qwen3-8B.json"


def test_the_gate_runs_the_real_capacity_call():
    from backend.arch.eval.precision import AS_SHIPPED
    from backend.arch.loader import SpecLoader
    from kernel_bench.sweep import _ceilings_for_deployment, routine_parallel_configs
    config = json.loads(SNAPSHOT.read_text())["config"]
    # The sweep resolves the spec once and threads it — matching no longer
    # scores a config without a model_id, so the gate must be handed the spec
    # the same way `build_sweep_jobs` hands it (its single entry point).
    resolved = SpecLoader().match(config, model_id="Qwen/Qwen3-8B")
    _label, parallel = routine_parallel_configs(config, resolved=resolved)[0]
    by_seq = _ceilings_for_deployment(
        config=config, gpu="H100-SXM", parallel=parallel, seq_lens=[1024],
        avg_output_tokens=256,
        # The gate states its precision, as `build_sweep_jobs` does (as
        # shipped when the caller names none).
        precision=AS_SHIPPED, resolved=resolved, framework="vllm")
    assert set(by_seq) == {1024}
    assert by_seq[1024].max_batch > 0


def test_the_gate_asks_the_framework_the_sweep_plans_under():
    """The gate's framework is the caller's, not the default.

    A binding and a card must agree on the device before either prices
    anything (`FrameworkBinding.deployed`). Left unstated, this gate asked
    the DEFAULT framework about whatever card it was given, so every cell of
    a non-CUDA box came back a gate error and the sweep wrote zero jobs —
    for a card whose own framework fits it fine.
    """
    import pytest

    from backend.arch.eval.precision import AS_SHIPPED
    from backend.arch.loader import SpecLoader
    from backend.inference.framework_bindings import FrameworkNotOnDevice
    from kernel_bench.sweep import _ceilings_for_deployment, routine_parallel_configs

    config = json.loads(SNAPSHOT.read_text())["config"]
    resolved = SpecLoader().match(config, model_id="Qwen/Qwen3-8B")
    _label, parallel = routine_parallel_configs(config, resolved=resolved)[0]

    def gate(framework: str):
        return _ceilings_for_deployment(
            config=config, gpu="Ascend-910C-A3", parallel=parallel,
            seq_lens=[1024], avg_output_tokens=256, precision=AS_SHIPPED,
            resolved=resolved, framework=framework)

    # The NPU card's own framework fits it...
    assert gate("vllm_ascend")[1024].max_batch > 0
    # ...and a CUDA framework on it is refused rather than priced, which is
    # what the gate used to ask by default.
    with pytest.raises(FrameworkNotOnDevice):
        gate("vllm")
