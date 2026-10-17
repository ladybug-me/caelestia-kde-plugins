"""Static risk classification for suggested commands.

The engine never runs anything; this module exists so that (a) the schema
lint can verify every rule declares a risk tier at least as severe as what
the command text implies, and (b) the renderer can label each step so a
human knows what they are about to copy-paste.

Severity ladder (ordered):
    READ_ONLY < STATE_CHANGING < PRIVILEGED < DESTRUCTIVE

Classification policy (deliberately conservative):
- Literal catastrophic shapes (mkfs, dd to a device, fork bombs, curl|sh,
  chmod 777 /, shred, wipefs) are DESTRUCTIVE.
- rm: recursive deletion is DESTRUCTIVE unless EVERY target is scoped to a
  regenerable location (~/.cache, $XDG_CACHE_HOME, $XDG_RUNTIME_DIR);
  non-recursive rm is STATE_CHANGING.
- sudo/pkexec/package-removal/service-disable/killall are PRIVILEGED.
- Known read-only verbs and --version/--status probes are READ_ONLY.
- Anything unrecognised defaults to STATE_CHANGING, never to READ_ONLY.
"""

from __future__ import annotations

import re
from typing import Optional

RISK_ORDER = {
    "READ_ONLY": 0,
    "STATE_CHANGING": 1,
    "PRIVILEGED": 2,
    "DESTRUCTIVE": 3,
}

# Deny-list seeds drawn from the real rule corpus (Task 3-b inventory) and
# docs/TROUBLESHOOTING.md.
_DESTRUCTIVE_LITERALS = [
    r"mkfs(\.\w+)?\s",
    r"dd\s+[^\n]*of=/dev/",
    r":\(\)\s*\{[^}]*\}\s*;\s*:",  # fork bomb
    r"chmod\s+(-R\s+)?777\s+/(?:\s|$)",
    r"curl[^|]*\|\s*(sudo\s+)?(ba)?sh(\s|$)",
    r"wget[^|]*\|\s*(sudo\s+)?(ba)?sh(\s|$)",
    r">\s*/dev/(sd|nvme|hd)",
    r">\s*/dev/mem",
    r"shred\s",
    r"wipefs\s",
    r"--no-preserve-root",
    # find's -delete performs recursive unlinking; must never classify as
    # READ_ONLY even though bare `find` is on the read-only list.
    r"find\s[^\n;|]*\s-delete\b",
]

# Anchor shared by the PRIVILEGED / STATE_CHANGING lists: the verb must not
# be glued to a word character, but any other preceding character (backtick,
# quote, "$(", separator) still counts. The old (^|\s|;|&&|\|\|) anchor let
# `sudo ...` / $(rm ...) dodge classification. Over-matching is the
# acceptable failure mode. The READ_ONLY list deliberately keeps its STRICT
# anchor: loosening it would widen READ_ONLY, the unsafe direction.
_VERB_ANCHOR = r"(?:^|[^\w-])"

_PRIVILEGED = [
    _VERB_ANCHOR + r"(sudo|pkexec|doas)\s",
    _VERB_ANCHOR + r"pacman\s+(-R|-Rs|-Rdd|-S(y|c|yy))",
    _VERB_ANCHOR + r"(dnf|apt|apt-get)\s+(install|remove|purge|upgrade|update|autoremove)",
    _VERB_ANCHOR + r"paru\s+-[RS]",
    _VERB_ANCHOR + r"gpasswd\s",
    _VERB_ANCHOR + r"usermod\s",
    _VERB_ANCHOR + r"systemctl\s+(--user\s+|--system\s+|\S+\s+)?disable",
    _VERB_ANCHOR + r"killall\s",
    _VERB_ANCHOR + r"pkill\s(-9|\s)",
]

_READ_ONLY = [
    r"(^|\s|;|&&|\|\|)(cat|ls|head|tail|grep|rg|stat|file|wc)\s",
    r"(^|\s|;|&&|\|\|)(du|df)\s",
    r"(^|\s|;|&&|\|\|)find\s",
    r"(^|\s|;|&&|\|\|)journalctl\s",
    r"(^|\s|;|&&|\|\|)systemctl\s+(--user\s+)?(status|is-active|is-enabled|show|list-unit-files)",
    r"(^|\s|;|&&|\|\|)kreadconfig\d?\s",
    r"(^|\s|;|&&|\|\|)pgrep\s",
    r"(^|\s|;|&&|\|\|)systemd-cgtop\s",
    r"(^|\s|;|&&|\|\|)qdbus6?\s+[^\n]*Introspect",
    r"git(\s+-C\s+\S+)?\s+(status|log|diff|ls-remote|branch|remote)",
    r"(^|\s|;|&&|\|\|)(pacman|dnf|apt|apt-get|paru)\s+(-Q|-Qi|-Qs|-Ss|search|list|show|info)",
    r"(^|\s|;|&&|\|\|)echo\s",
    r"(^|\s|;|&&|\|\|)env(\s|$)",
    r"--version(\s*$|\s*&&)",  # any tool's --version probe
    # Repo diagnostic scripts: verified read-only checkers.
    r"bash\s+\S*(check-workspace-tracker|list-plugins|menu_perf_stats)\.sh",
    # Per-card GPU read loop from the #616 rule.
    r"for\s+f\s+in\s+/sys/class/drm",
    # Reading shell state for reports.
    r"du\s+-[a-zA-Z]*a",
]

_STATE_CHANGING = [
    _VERB_ANCHOR + r"kwriteconfig\d?\s",
    _VERB_ANCHOR + r"systemctl\s+(--user\s+)?(restart|start|stop|enable)",
    _VERB_ANCHOR + r"qdbus6?\s+org\.kde\.KWin\s+/KWin\s+reconfigure",
    _VERB_ANCHOR + r"kbuildsycoca\d?\s",
    _VERB_ANCHOR + r"lookandfeeltool\s+--apply",
    _VERB_ANCHOR + r"rm\s",
    _VERB_ANCHOR + r"sed\s+[^|\n]*-i",
    _VERB_ANCHOR + r"kill\s",
    _VERB_ANCHOR + r"git\s+(stash|checkout|reset|clean|submodule)",
    _VERB_ANCHOR + r"bash\s+(update\.sh|scripts/setup\.sh|scripts/|~/.config/quickshell)",
    _VERB_ANCHOR + r"\$EDITOR\s",
    _VERB_ANCHOR + r"(nano|vim|nvim|kate|micro)\s",
    _VERB_ANCHOR + r"ccache\s+-C",
]

# rm targets scoped to these prefixes are regenerable caches / volatile
# runtime state: recursive deletion there is STATE_CHANGING, not DESTRUCTIVE.
# Stored lowercase and compared against target.lower() so odd spellings
# ("RM", "${xdg_runtime_dir") cannot dodge the scope check.
_CACHE_SCOPED_PREFIXES = tuple(
    prefix.lower()
    for prefix in (
        "~/.cache/",
        "$XDG_CACHE_HOME/",
        "${XDG_CACHE_HOME",
        "$XDG_RUNTIME_DIR/",
        "${XDG_RUNTIME_DIR",
        "/tmp/",
    )
)


def _classify_rm(text: str) -> Optional[str]:
    """Classify a command that contains rm; None if there is no rm.

    The anchor matches rm after start/whitespace/;/&/|/backtick/"$(" /quote so
    neither a prefix word ("then rm -rf /") nor punctuation wrapping
    ("`rm -rf /`", "$(rm -rf /etc)") can dodge classification. Matching is
    case-insensitive: "RM -RF /" and "rm -Rf" (uppercase -R IS recursive)
    must not slip through. Over-classification is the acceptable failure
    mode; under-classification is not.
    """
    if not re.search(r"(?:^|[^\w-])rm\s", text, re.IGNORECASE):
        return None
    for pattern in _DESTRUCTIVE_LITERALS:
        if re.search(pattern, text, re.IGNORECASE):
            return "DESTRUCTIVE"
    # Collect rm invocations with their flags and targets.
    for match in re.finditer(r"(?:^|[^\w-]\s*)rm\s+([^\n]+)", text, re.IGNORECASE):
        rest = match.group(1)
        tokens = rest.strip().split()
        flags = "".join(tok for tok in tokens if tok.startswith("-"))
        targets = [tok.strip("\"'`") for tok in tokens if not tok.startswith("-")]
        recursive = "r" in flags.lower()  # -r AND -R are both recursive
        if not recursive:
            continue
        if targets and all(
            any(t.lower().startswith(prefix) for prefix in _CACHE_SCOPED_PREFIXES)
            for t in targets
        ):
            return "STATE_CHANGING"  # scoped to regenerable cache/runtime state
        return "DESTRUCTIVE"
    return "STATE_CHANGING"  # plain non-recursive rm


def classify(command: str) -> str:
    """Return the most severe risk tier the command text implies.

    All matching is case-insensitive ("RM -RF /" expresses the same intent
    as "rm -rf /"; missing a destructive shape because of letter case would
    be under-classification). Terminal escape sequences are stripped first so
    ANSI-wrapped flags cannot dodge the rm analysis. The most severe
    matching tier wins.
    """
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", command.strip())

    rm_risk = _classify_rm(text)
    if rm_risk == "DESTRUCTIVE":
        return "DESTRUCTIVE"

    for pattern in _DESTRUCTIVE_LITERALS:
        if re.search(pattern, text, re.IGNORECASE):
            return "DESTRUCTIVE"
    for pattern in _PRIVILEGED:
        if re.search(pattern, text, re.IGNORECASE):
            return "PRIVILEGED"
    if rm_risk == "STATE_CHANGING":
        return "STATE_CHANGING"
    for pattern in _READ_ONLY:
        if re.search(pattern, text, re.IGNORECASE):
            return "READ_ONLY"
    for pattern in _STATE_CHANGING:
        if re.search(pattern, text, re.IGNORECASE):
            return "STATE_CHANGING"
    # Conservative default: anything unrecognised is treated as state
    # changing, never as safe.
    return "STATE_CHANGING"


def at_least_as_severe(declared: str, classified: str) -> bool:
    """True when the declared tier is >= the classified tier."""
    return RISK_ORDER[declared] >= RISK_ORDER[classified]
