"""Manifest for the diagnostics capability: permissions, risk tier, budgets."""
from assistant.capabilities._registry import CapManifest

MANIFEST = CapManifest(
    name="diagnostics",
    permissions=("fs:read", "rules:read",),
    risk_tier="read_only",
    latency_budget_ms=400,
    memory_budget_mb=40,
)
