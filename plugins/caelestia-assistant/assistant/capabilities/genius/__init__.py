"""assistant.capabilities.genius — the universal local intelligence layer (no LLM).

15 algorithm modules + a universal task router, stdlib-only, injected
into the caelestia-assistant safety spine (inert suggestions, ledger
proposals, honest verdicts). See genius/README.md for the full map.
"""
# ONE version source: assistant/__init__.py. This re-export keeps
# ``assistant.capabilities.genius.__version__`` working without a second copy of the
# number that can drift from the package's own.
from assistant import __version__  # noqa: F401
