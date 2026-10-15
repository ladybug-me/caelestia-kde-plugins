"""assistant.capabilities.graph — the shell knowledge graph (exponential-build-5 F12).

Public API (see build.py / queries.py docstrings for contracts):

  build_graph(ledger_path=None)  — deterministic, provenance-carrying build
  graph_hash(graph)              — rebuild-drift detector
  dump_canonical(graph)          — canonical JSON text
  what_affects / what_breaks / explanation_path / related /
  personalized_pagerank          — the query surface

CLI: ``caelestia-assist graph build|affects|breaks|path|related|rank``.
"""

from .build import build_graph, dump_canonical, graph_hash
from .queries import (explanation_path, personalized_pagerank, related,
                      resolve_config, what_affects, what_breaks)

__all__ = ["build_graph", "graph_hash", "dump_canonical", "what_affects",
           "what_breaks", "explanation_path", "related",
           "personalized_pagerank", "resolve_config"]
