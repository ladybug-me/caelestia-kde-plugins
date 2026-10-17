"""assistant.capabilities.agent — the agentic orchestrator (goal -> graph -> consent -> act -> learn).

Issue #120's propose/approve loop, generalised to EVERY layer:

    assistant.goals      HTN-style decomposition into a dependency-ordered
                         task graph over the layers (diagnose, retrieve,
                         route, tidy, brief, genius)
    assistant.clarify    20-questions dialogue: the clarifying question with
                         the highest expected information gain, or none
    assistant.engine     simulate (project, touch nothing) then execute with
                         a per-node consent gate; outcomes feed the learners

Safety contract: unchanged. Every STATE_CHANGING node needs a True from the
caller's consent function; refused nodes skip their dependents; there is no
PRIVILEGED or DESTRUCTIVE node at all — such work stays an inert
SUGGESTED_NOT_EXECUTED suggestion inside a read-only result.
"""
from . import goals, clarify, engine
from .engine import Agent

__all__ = ["goals", "clarify", "engine", "Agent"]
