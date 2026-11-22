"""Source-specific serving is explicit in the corpus deployment."""
from scripts.dump_sol_surface import deployments_for


def test_kimi_dcp8_uses_the_audited_framework():
    cases = deployments_for("moonshotai__Kimi-K3.1-Pro.json")
    assert cases["b300_dcp8"]["inputs"]["framework"] == "sglang"
    assert cases["b300_dcp8"]["parallel"]["dcp"] == 8
    assert "inputs" not in cases["eight"]
    assert "b300_dcp8" not in deployments_for("moonshotai__Kimi-K3.json")
