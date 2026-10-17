"""The cortex layer: learned intelligence for the assistant, no LLM.

Package map (each module keeps its own guarantees docstring):

- ``lexicon``    morphology + fuzzy matching + the domain lexicon (pure)
- ``corpus``     deterministic synthetic paraphrase corpus (pure)
- ``vectorize``  BM25+ index and PPMI/random-projection embeddings (seeded)
- ``router``     the universal intent router over every surface (pure)
- ``compound``   multi-intent clause splitting (pure)
- ``nlhistory``  natural-language undo/history queries (pure planning)
- ``session``    conversational state: anaphora, ellipsis, clarification
- ``memory``     episodic interaction memory with forgetting curve
- ``learn``      online learning loop (AdaGrad logistic, calibration,
                 bandit over routing strategies, drift monitor)
- ``pipeline``   the orchestrator wiring cortex into the settings spine

Design spine (inherited from the rest of the assistant): every learned
component ranks and proposes; the settings planner/applier and the brain
ledger stay the only write paths; honest verdicts (AMBIGUOUS beats a
silent guess) everywhere; stdlib only.
"""
