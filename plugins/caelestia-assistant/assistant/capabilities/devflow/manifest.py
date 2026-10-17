"""Manifest for the devflow capability: permissions, risk tier, budgets."""
from assistant.capabilities._registry import CapManifest

MANIFEST = CapManifest(
    name="devflow",
    permissions=("fs:read", "git:read",),
    risk_tier="read_only",
    latency_budget_ms=500,
    memory_budget_mb=40,
)
