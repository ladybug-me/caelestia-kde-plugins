"""Manifest for the scan capability: permissions, risk tier, budgets."""
from assistant.capabilities._registry import CapManifest

MANIFEST = CapManifest(
    name="scan",
    permissions=("fs:read-one-pass",),
    risk_tier="read_only",
    latency_budget_ms=400,
    memory_budget_mb=40,
)
