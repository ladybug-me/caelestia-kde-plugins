"""Layer 4: issue-DRAFTING helper (drafts only — never files anything).

This layer composes clean, paste-ready issue text from a described problem
and STOPS THERE. Guarantees, same safety box as the rest of the assistant:

- No network code of any kind (no http/urllib/socket imports; the whole-
  assistant import scan in assistant/diagnostics/schema_lint.py enforces it).
- Drafts are written ONLY to local files, and ONLY when the CLI is invoked
  with --confirm. The default is preview-only: the draft is printed to
  stdout and nothing touches the filesystem.
- Nothing is ever posted, submitted, opened, or transmitted. The user
  copy-pastes the draft into GitHub's new-issue form themselves.

Scope note (from the maintainer threads captured in the retrieval corpus,
ISS-120 / ISS-802): the assistant's job is "issue reportings to repo
directly" as DRAFTING ONLY — this layer implements exactly that boundary.
"""

from pathlib import Path

ISSUES_DIR = Path(__file__).resolve().parent
REPO_ROOT = ISSUES_DIR.parent.parent
TEMPLATES_DIR = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"
BUG_TEMPLATE_FILE = TEMPLATES_DIR / "1-issue.yml"
FEATURE_TEMPLATE_FILE = TEMPLATES_DIR / "2-feature_request.yml"

__all__ = [
    "BUG_TEMPLATE_FILE",
    "FEATURE_TEMPLATE_FILE",
    "ISSUES_DIR",
    "REPO_ROOT",
    "TEMPLATES_DIR",
]
