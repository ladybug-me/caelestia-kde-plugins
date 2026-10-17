"""Manifest for the issues capability: permissions, risk tier, budgets."""
from assistant.capabilities._registry import CapManifest

MANIFEST = CapManifest(
    name="issues",
    permissions=("fs:write-drafts-only", "fs:read",),
    risk_tier="journaled",
    latency_budget_ms=500,
    memory_budget_mb=40,
)
