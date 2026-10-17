"""scan.scanner — the one-pass stream scanner composing every structure.

    from assistant.capabilities.scan import Automaton, scan_stream
    auto = Automaton(["quickshell has crashed", "pragma", "signal handoff"])
    summary = scan_stream(lines, auto)

`scan_stream` walks the line iterable exactly once and returns a plain dict:
per-pattern hit counts, unique-line tracking (Bloom), distinct-line estimate
(HLL), token frequency sketch (Count-Min), a reservoir of example lines, and
the EWMA + Page-Hinkley rate-drift alarm over fixed-size chunks. For the QML
bridge the same dict is what `{"op":"scan_text"}` returns.

Nothing is executed, nothing is stored beyond the bounded structures, and a
2 GB log and a 2 KB log cost the same RAM.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .ac import Automaton
from .bloom import BloomFilter
from .sketch import (CountMinSketch, HyperLogLog, NgramDivergence,
                     Reservoir, EWMA, page_hinkley)

__all__ = ["scan_stream", "scan_text", "render"]

_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is",
         "at", "by", "with", "from", "was", "were", "be", "this", "that",
         "it", "as", "not", "but", "are", "into", "my"}


def scan_stream(lines: Iterable[str], automaton: Automaton,
                chunk_size: int = 500, reservoir_k: int = 40,
                bloom_capacity: int = 200_000,
                token_top_candidates: Optional[List[str]] = None,
                novelty: bool = False, novelty_block_lines: int = 500,
                ) -> Dict[str, Any]:
    """One pass over `lines`; the returned summary is JSON-serialisable.

    ``novelty=True`` (opt-in, B1) adds a "novelty" section: the windowed
    KL divergence of the log's token n-gram distribution per block against
    the running baseline (previous block + the scanner's Count-Min), plus
    a Page-Hinkley alarm over the KL series. Still one pass, still bounded
    RAM (two block-sized dicts); the default summary is unchanged."""
    bloom = BloomFilter(capacity=bloom_capacity)
    hll = HyperLogLog(precision=12)
    cms = CountMinSketch(epsilon=0.002, delta=0.001)
    reservoir = Reservoir(k=reservoir_k)
    ewma = EWMA(alpha=0.05)
    divergence = NgramDivergence(block_lines=novelty_block_lines) if novelty else None
    novelty_blocks: List[Dict[str, Any]] = []

    hits_by_pattern: Dict[int, int] = {}
    lines_total = 0
    matching = 0
    unique_matching = 0
    chunk_counts: List[int] = []
    cur_chunk = 0
    tokens: List[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        lines_total += 1
        if divergence is not None:
            report = divergence.offer(line)
            if report is not None:
                novelty_blocks.append(report)
        hits = automaton.scan(line)
        cur_chunk += len(hits)
        for _, pid in hits:
            hits_by_pattern[pid] = hits_by_pattern.get(pid, 0) + 1
        if hits:
            matching += 1
            reservoir.offer(line)
            norm = line.strip().lower()
            hll.add(norm)
            if bloom.add_if_absent(norm):
                unique_matching += 1
            for tok in norm.replace(":", " ").replace(",", " ").split():
                if tok not in _STOP and len(tok) > 2 and not tok.isdigit():
                    cms.add(tok)
                    tokens.append(tok)
        if lines_total % chunk_size == 0:
            chunk_counts.append(cur_chunk)
            ewma.update(cur_chunk / chunk_size)
            cur_chunk = 0

    if lines_total % chunk_size:
        chunk_counts.append(cur_chunk)
        ewma.update(cur_chunk / max(1, lines_total % chunk_size))

    drift = page_hinkley([c / float(chunk_size) for c in chunk_counts],
                         threshold=4.0, min_instances=10)
    top: List[List[Any]] = []
    if token_top_candidates:
        for cnt, tok in cms.top_candidates(token_top_candidates, k=12):
            top.append([tok, cnt])

    summary = {
        "lines_scanned": lines_total,
        "lines_matching": matching,
        "unique_matching_estimate": round(hll.count()),
        "unique_matching_exact_new": unique_matching,
        "pattern_hits": [
            {"pattern": automaton.pattern(pid), "hits": n}
            for pid, n in sorted(hits_by_pattern.items(), key=lambda kv: -kv[1])
        ],
        "top_tokens": top,
        "examples": reservoir.sample,
        "rate": {
            "ewma_per_line": round(ewma.mean, 6) if ewma.mean is not None else None,
            "page_hinkley_change_at_chunk": drift["change_at"],
            "chunks": len(chunk_counts),
        },
        "state": {
            "bloom": bloom.to_dict(),
            "hll": hll.to_dict(),
            "cms": cms.to_dict(),
            "reservoir": reservoir.to_dict(),
            "ewma": ewma.to_dict(),
        },
    }
    if divergence is not None:
        summary["novelty"] = {
            "blocks": novelty_blocks,
            "last_vs_cms": divergence.kl_vs_cms(),
            "alarm": divergence.alarm(),
        }
    return summary


def scan_text(text: str, automaton: Automaton, **kwargs: Any) -> Dict[str, Any]:
    return scan_stream(text.splitlines(), automaton, **kwargs)


# ---------------------------------------------------------------------------
def render(summary: Dict[str, Any]) -> str:
    """Human renderer — mirrors the other layers' plain-text style."""
    out: List[str] = []
    n = summary["lines_scanned"]
    m = summary["lines_matching"]
    out.append(f"scanned {n} lines; {m} matched known signatures "
               f"({summary['unique_matching_estimate']} distinct by HLL estimate)")
    hits = summary["pattern_hits"]
    if hits:
        out.append("signature hits:")
        for h in hits[:10]:
            out.append(f"  {h['hits']:>6}  {h['pattern']}")
    if summary["top_tokens"]:
        toks = ", ".join(f"{t}({c})" for t, c in summary["top_tokens"][:8])
        out.append(f"dominant tokens (Count-Min): {toks}")
    rate = summary["rate"]
    if rate["page_hinkley_change_at_chunk"] is not None:
        out.append(f"RATE CHANGE: Page-Hinkley alarmed at chunk "
                   f"{rate['page_hinkley_change_at_chunk']} — the error rate "
                   f"shifted partway through this stream")
    if summary["examples"]:
        out.append("sampled matching lines (reservoir):")
        for line in summary["examples"][:5]:
            shown = line if len(line) <= 110 else line[:107] + "..."
            out.append(f"  {shown}")
    if not hits:
        out.append("no known signature matched; run: caelestia-assist diagnose "
                   "with a pasted excerpt for the full rule engine")
    return "\n".join(out)
