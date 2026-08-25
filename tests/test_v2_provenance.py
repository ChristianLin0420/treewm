"""V2 lifecycle provenance invariants."""

from treewm.utils import provenance as provenance_module


def test_runtime_identity_is_stable_across_slurm_nodes(monkeypatch):
    """Kernel/host metadata is recorded, but must not invalidate exact resume."""
    monkeypatch.setattr(provenance_module.platform, "platform", lambda: "node-a-kernel")
    monkeypatch.setattr(provenance_module.platform, "machine", lambda: "node-a-machine")
    first = provenance_module.runtime_fingerprint()

    monkeypatch.setattr(provenance_module.platform, "platform", lambda: "node-b-kernel")
    monkeypatch.setattr(provenance_module.platform, "machine", lambda: "node-b-machine")
    second = provenance_module.runtime_fingerprint()

    assert first["sha256"] == second["sha256"]
    assert first["software"] == second["software"]
    assert first["host"] != second["host"]
