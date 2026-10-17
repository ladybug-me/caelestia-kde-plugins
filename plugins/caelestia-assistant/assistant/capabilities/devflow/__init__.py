"""assistant.capabilities.devflow — developer-workflow drafting tools.

SCOPE DISCLAIMER (the reason this is a separate top-level package):
this domain exists for the DEVELOPER'S OWN workflow — commit-message
drafting, PR-description skeletons, TODO/FIXME triage over source trees.
It has NO relationship to caelestia-kde issue #120 or the KDE shell, it
is NOT part of the assistant's shell-facing feature surface, and it must
never be included in any upstream-bound pull request under any framing.
If a #120-scoped task ever seems to need this package, that is a signal
the partition is wrong — stop and re-scope, don't cross it.

What it does, all deterministic and templated (NO generation, no model):

- diffstat.commit_message: a conventional-commit skeleton from
  ``git diff --numstat`` text — keyword-to-type classification
  (tests/docs/ci/feat/refactor/chore), mechanical scope from the common
  top-level directory, subject/body filled with the REAL file/line
  deltas. The human writes the actual prose; this is scaffolding.
- diffstat.pr_skeleton: a PR-description skeleton from the same stats,
  shaped like the upstream project's PR template (what/why, changes by
  area with real deltas, testing checklist, type checkboxes pre-filled
  by the same classification).
- todo.triage_tree: TODO/FIXME/XXX/HACK/BUG marker triage over a source
  tree, reusing the scan layer's Aho-Corasick automaton (built for logs,
  equally exact over source) — one pass per file, marker + line number +
  the comment's own text, grouped by marker.

Stdlib only like everything else; reads only the tree the caller names;
writes nothing, ever.
"""
