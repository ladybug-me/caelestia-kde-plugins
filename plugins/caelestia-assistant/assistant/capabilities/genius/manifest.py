"""Manifest for the genius capability: permissions, risk tier, budgets."""
from assistant.capabilities._registry import CapManifest

MANIFEST = CapManifest(
    name="genius",
    permissions=("compute:local",),
    risk_tier="read_only",
    latency_budget_ms=2000,
    memory_budget_mb=40,
)
