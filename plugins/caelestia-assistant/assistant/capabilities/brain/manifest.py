"""Manifest for the brain capability: permissions, risk tier, budgets."""
from assistant.capabilities._registry import CapManifest

MANIFEST = CapManifest(
    name="brain",
    permissions=("state:read-write", "fs:read",),
    risk_tier="journaled",
    latency_budget_ms=500,
    memory_budget_mb=40,
)
