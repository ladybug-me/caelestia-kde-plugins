"""Manifest for the dedupe capability."""
from assistant.capabilities._registry import CapManifest

MANIFEST = CapManifest(
    name="dedupe",
    permissions=("fs:read", "fs:quarantine-move-journaled"),
    risk_tier="journaled",
    latency_budget_ms=1500,
    memory_budget_mb=40,
)
