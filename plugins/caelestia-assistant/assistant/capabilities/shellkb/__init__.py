"""assistant.capabilities.shellkb — the shell employee (Groups B6-B10).

Everything here is READ-ONLY knowledge about the shell's own command
line, configs, and dependencies: it parses files (never executes
binaries), explains command lines token by token, answers "how do I"
questions from a self-authored cheat sheet, diffs and three-way-merges
structured JSON configs, and explains dependency conflicts PubGrub-style.

The posture is the same as the rest of the assistant, stated once:

- every command the shellkb prints is labeled SUGGESTED_NOT_EXECUTED —
  the assistant never runs a shell command and never will;
- proposals, never writes: config merges are printed, not applied
  (shell.json is only ever edited through the settings layer's own
  ``--apply`` gate);
- the source of truth for CLI syntax is the upstream FILES (usage
  heredocs, case dispatch, printf usage lines) parsed offline —
  "knows" means "read the manual", not "tried it";
- deterministic everywhere: sorted iteration, no RNG, no clock unless
  the caller passes one.
"""
