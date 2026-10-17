"""Manifest for the graph capability: permissions, risk tier, budgets."""
from assistant.capabilities._registry import CapManifest

MANIFEST = CapManifest(
    name="graph",
    permissions=("fs:read", "registry:read",),
    risk_tier="read_only",
    latency_budget_ms=800,
    memory_budget_mb=40,
)
