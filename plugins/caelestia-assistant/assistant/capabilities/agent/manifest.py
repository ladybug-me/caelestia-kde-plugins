"""Manifest for the agent capability: permissions, risk tier, budgets."""
from assistant.capabilities._registry import CapManifest

MANIFEST = CapManifest(
    name="agent",
    permissions=("probe:exec-read-only", "plan:simulate-first",),
    risk_tier="privileged",
    latency_budget_ms=5000,
    memory_budget_mb=40,
)
