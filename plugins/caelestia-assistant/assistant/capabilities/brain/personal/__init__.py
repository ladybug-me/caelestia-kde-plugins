"""assistant.capabilities.brain.personal — the opt-in personal-knowledge-management tools.

This package is NOT part of the caelestia-kde issue #120 feature surface.

Everything in here is a self-contained personal-productivity engine (markdown
vault organization, spaced repetition, task survival analysis, decision
journaling, day planning). It shares no data with the Caelestia shell: no
shell.json access, no settings-layer imports, no bridge ops, no subcommands
in `caelestia-assist` or `python3 -m assistant.capabilities.brain`.

It is exposed only through its own entry point:

    python3 -m assistant.capabilities.brain.personal --help

and ships in the same repository purely because it reuses the same stdlib-only
engines and the same "propose, never apply" ledger discipline. See
personal/README.md for the full separation statement.
"""
