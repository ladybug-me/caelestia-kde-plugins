"""brain — lean, non-LLM intelligence layer for the caelestia assistant.

Everything here is stdlib-only and deterministic given the same state.
Autonomous actions never execute: they become proposals in a ledger that the
user approves or rejects, and those decisions become training labels.
"""
