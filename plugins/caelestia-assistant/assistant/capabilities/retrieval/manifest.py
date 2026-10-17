"""Manifest for the retrieval capability: permissions, risk tier, budgets."""
from assistant.capabilities._registry import CapManifest

MANIFEST = CapManifest(
    name="retrieval",
    permissions=("fs:read", "index:read",),
    risk_tier="read_only",
    latency_budget_ms=300,
    memory_budget_mb=40,
)
