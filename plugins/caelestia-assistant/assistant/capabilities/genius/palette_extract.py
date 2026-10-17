"""Wallpaper-driven palette extraction + scheme accessibility audit
(issue #120 Phase 2.4 / 2.5). Colocated with the OKLab/WCAG code it
reuses: assistant/genius/creative.py owns the sRGB<->OKLab math and the
WCAG contrast ratio; this module adds the extraction and audit layers on
top and imports them — no duplicated color science.

1. Wallpaper -> candidate accent (Phase 2.4), every step named:

   - k-means color quantization in OKLab space (uniform steps in a
     perceptual color space; reuses genius/data.py::kmeans with k-means++
     seeding, the same deterministic implementation the genius layer
     ships and tests). Pixels come from a minimal stdlib PNG reader
     (zlib + struct only — no new imports) with deterministic stride
     sampling bounded by max_pixels.
   - Accent selection: among clusters with pixel share >= share_floor,
     the one with the highest chroma (C = sqrt(a^2 + b^2) in OKLab).
     Rationale, stated plainly: a wallpaper's largest cluster is usually
     its background (near-neutral), and an accent needs chroma; share
     filtering keeps the pick honest to the image instead of harvesting
     the one saturated pixel.
   - WCAG verdicts for the candidate against white and black via
     creative.contrast_ratio — surfaced, never auto-applied.

   Surfacing (Phase 2.4) rides the EXISTING inert-scheme-suggestion path:
   settings/cli.py --wallpaper-palette feeds the result to the same
   SUGGESTED verdict / SUGGESTED_NOT_EXECUTED renderer the parser uses for
   accent-color requests. No new suggestion mechanism is created.

2. Scheme accessibility audit (Phase 2.5):

   - All pairwise text/background contrast in the scheme (WCAG 2.x via
     creative.contrast_ratio), failing pairs flagged at the 4.5:1 AA
     normal-text threshold.
   - Protanopia/deuteranopia/tritanopia simulation: the Viénot-Brettel-
     Mollon (1999) 3x3 linear-RGB transform matrices — the standard
     dichromacy simulation, applied in LINEAR sRGB (not gamma-encoded)
     as that family of papers requires.
   - Nearest compliant color: bounded bisection on the OKLab L axis
     (hue/chroma preserved) until contrast against the background meets
     the target — a bounded, deterministic repair, reported with the
     achieved ratio. If no L reaches the target within the bounds, the
     honest answer is that the hue cannot reach compliance on that
     background.

Read-only: no file writes, no scheme changes — ever.
"""

from __future__ import annotations

import math
import struct
import zlib
from typing import Any, Dict, List, Sequence, Tuple

from .creative import (_linear_to_srgb, _srgb_to_linear, contrast_ratio,
                       hex_to_rgb, oklab_to_rgb, rgb_to_oklab, rgb_to_hex)

# --- Viénot, Brettel & Mollon (1999) dichromacy simulation, linear RGB ---
_CB_MATRICES = {
    "protanopia": (
        0.11238, 0.88762, 0.0,
        0.11238, 0.88762, 0.0,
        0.00401, -0.00401, 1.0,
    ),
    "deuteranopia": (
        0.29275, 0.70725, 0.0,
        0.29275, 0.70725, 0.0,
        -0.02234, 0.02234, 1.0,
    ),
    "tritanopia": (
        1.0, 0.14461, -0.14461,
        0.0, 0.85924, 0.14076,
        0.0, 0.85924, 0.14076,
    ),
}
CONTRAST_AA_NORMAL = 4.5
SHARE_FLOOR = 0.15


# ---------------------------------------------------------------------------
# Minimal PNG reader (8-bit RGB / RGBA, non-interlaced) — stdlib only.
# ---------------------------------------------------------------------------

def png_grid(data: bytes) -> Tuple[int, int, int, List[bytes]]:
    """Decode an 8-bit truecolor (RGB/RGBA), non-interlaced PNG to its
    full row grid: ``(width, height, channels, rows)`` where each row is
    the defiltered scanline bytes. The ONE decoder in this codebase —
    ``png_pixels`` below and the screenshot structural diff both call
    it; no second PNG reader exists.

    Supports the 8-bit truecolor variants wallpapers and screenshots
    actually ship as (color type 2 RGB and 6 RGBA, non-interlaced).
    Anything else raises ValueError — honestly, not by guessing.
    """
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos = 8
    width = height = bit_depth = color_type = interlace = None
    idat = bytearray()
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        chunk_type = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filt, interlace = \
                struct.unpack(">IIBBBBB", payload)
        elif chunk_type == b"IDAT":
            idat.extend(payload)
        elif chunk_type == b"IEND":
            break
        pos += 12 + length
    if not width or not height:
        raise ValueError("PNG header missing or incomplete")
    if bit_depth != 8:
        raise ValueError(f"unsupported bit depth {bit_depth} (only 8)")
    if color_type not in (2, 6):
        raise ValueError(f"unsupported color type {color_type} (only 2=RGB, 6=RGBA)")
    if interlace != 0:
        raise ValueError("interlaced PNGs are not supported")

    channels = 3 if color_type == 2 else 4
    stride = width * channels
    raw = zlib.decompress(bytes(idat))
    rows: List[bytes] = []
    prev = bytearray(stride)
    offset = 0
    for _y in range(height):
        if offset >= len(raw):
            break
        ftype = raw[offset]
        offset += 1
        line = bytearray(raw[offset:offset + stride])
        offset += stride
        if ftype == 1:  # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        elif ftype != 0:
            raise ValueError(f"unknown PNG filter type {ftype}")
        prev = line
        rows.append(bytes(line))

    return width, height, channels, rows


def png_pixels(data: bytes, max_pixels: int = 40000) -> List[Tuple[int, int, int]]:
    """Decode a PNG to a deterministically sampled list of RGB pixels.

    Sampling is a fixed stride over the scanline order (via ``png_grid``,
    the one decoder) so the result is stable for a given file.
    """
    width, height, channels, rows = png_grid(data)
    total = width * height
    step = max(1, total // max_pixels) if total > max_pixels else 1
    pixels: List[Tuple[int, int, int]] = []
    for idx in range(0, total, step):
        y, x = divmod(idx, width)
        base = x * channels
        line = rows[y]
        r, g, b = line[base], line[base + 1], line[base + 2]
        pixels.append((r, g, b))
    return pixels


# ---------------------------------------------------------------------------
# Phase 2.4: k-means quantization in OKLab -> WCAG-checked accent candidate.
# ---------------------------------------------------------------------------

def extract_accent(pixels: Sequence[Tuple[int, int, int]],
                   k: int = 6, share_floor: float = SHARE_FLOOR,
                   seed: int = 42) -> Dict[str, Any]:
    """k-means quantization in OKLab; pick the accent-like cluster.

    Returns {"accent", "clusters": [{"hex", "share", "chroma"}...],
    "contrast": {on_white, on_black verdicts}, "selection"}.
    """
    from assistant.capabilities.genius.data import kmeans  # local: genius layer owns k-means

    if not pixels:
        raise ValueError("no pixels to cluster")
    pts = [list(rgb_to_oklab(tuple(v / 255.0 for v in px))) for px in pixels]
    n = len(pts)
    k = max(2, min(k, n))
    result = kmeans(pts, k, seed=seed)
    labels = result["labels"]
    centers = result["centers"]

    clusters: List[Dict[str, Any]] = []
    for cid in range(k):
        count = labels.count(cid)
        if count == 0:
            continue
        L, a, b = centers[cid]
        chroma = math.sqrt(a * a + b * b)
        clusters.append({
            "hex": rgb_to_hex(*oklab_to_rgb([L, a, b])),
            "share": round(count / n, 4),
            "chroma": round(chroma, 5),
            "L": round(L, 4),
        })
    clusters.sort(key=lambda c: -c["share"])

    eligible = [c for c in clusters if c["share"] >= share_floor]
    if eligible:
        best = max(eligible, key=lambda c: c["chroma"])
        selection = (f"highest-chroma cluster among {len(eligible)} with "
                     f"share >= {share_floor}")
    else:
        best = clusters[0]
        selection = ("no cluster met the share floor; fell back to the "
                     "largest cluster")

    on_white = contrast_ratio(best["hex"], "#ffffff")
    on_black = contrast_ratio(best["hex"], "#000000")
    return {
        "accent": best["hex"],
        "clusters": clusters,
        "contrast": {"on_white": on_white, "on_black": on_black},
        "selection": selection,
        "n_pixels": n,
    }


def accent_from_png(data: bytes, max_pixels: int = 40000,
                    k: int = 6, seed: int = 42) -> Dict[str, Any]:
    """PNG bytes -> WCAG-checked accent candidate (the 2.4 one-call)."""
    pixels = png_pixels(data, max_pixels=max_pixels)
    result = extract_accent(pixels, k=k, seed=seed)
    result["source"] = f"png ({len(pixels)} sampled pixels)"
    return result


# ---------------------------------------------------------------------------
# Phase 2.5: scheme accessibility audit + dichromacy simulation + repair.
# ---------------------------------------------------------------------------

def simulate_colorblind(hex_color: str, kind: str) -> str:
    """One color through the Viénot-Brettel-Mollon dichromacy transform."""
    if kind not in _CB_MATRICES:
        raise ValueError(f"unknown simulation {kind!r}; "
                         f"known: {', '.join(sorted(_CB_MATRICES))}")
    rgb = [int(hex_color.lstrip("#")[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linear = [_srgb_to_linear(v) for v in rgb]
    m = _CB_MATRICES[kind]
    out = [
        m[0] * linear[0] + m[1] * linear[1] + m[2] * linear[2],
        m[3] * linear[0] + m[4] * linear[1] + m[5] * linear[2],
        m[6] * linear[0] + m[7] * linear[1] + m[8] * linear[2],
    ]
    clamped = [min(1.0, max(0.0, v)) for v in out]
    return rgb_to_hex(*[_linear_to_srgb(v) for v in clamped])


def audit_scheme(scheme: Dict[str, str], simulate: bool = True) -> Dict[str, Any]:
    """All pairwise text/background contrast in the scheme, plus optional
    dichromacy simulation of every role color.

    scheme: {"role": "#hex"}; text-ish roles are the _TEXT_ROLES set,
    backgrounds the _BG_ROLES set (same convention sanity.py uses).
    Returns {"pairs": [...], "failing": [...], "simulated": {...}}.
    """
    text_roles = ("foreground", "text", "accent")
    bg_roles = ("background", "base", "surface")
    pairs: List[Dict[str, Any]] = []
    for tr in text_roles:
        fg = scheme.get(tr)
        if fg is None:
            continue
        for br in bg_roles:
            bg = scheme.get(br)
            if bg is None:
                continue
            v = contrast_ratio(str(fg), str(bg))
            pairs.append({"text": tr, "background": br,
                          "ratio": v["ratio"], "aa_normal_text": v["aa_normal_text"],
                          "aaa_normal_text": v["aaa_normal_text"]})
    failing = [p for p in pairs if p["ratio"] < CONTRAST_AA_NORMAL]
    simulated: Dict[str, Any] = {}
    if simulate:
        for role, hexv in scheme.items():
            simulated[role] = {k: simulate_colorblind(str(hexv), k)
                               for k in sorted(_CB_MATRICES)}
    return {"pairs": pairs, "failing": failing, "simulated": simulated}


def nearest_compliant(hex_color: str, background: str,
                      target: float = CONTRAST_AA_NORMAL,
                      steps: int = 40) -> Dict[str, Any]:
    """Bounded bisection on the OKLab L axis toward WCAG compliance.

    Hue and chroma are preserved. WCAG relative luminance is monotone in
    linear luminance, OKLab L is monotone in linear luminance, so walking
    L away from the background's lightness monotonically increases the
    contrast ratio: ratio(t) along the segment L -> L_extreme is monotone,
    and classic bisection finds the MINIMAL lightness move that meets the
    target (deterministic; the extreme itself is the honest fallback).

    Returns {"ok", "hex", "ratio", "note"} — ok=False honestly reports
    that this hue cannot reach the target on that background.
    """
    L, a, b = rgb_to_oklab(hex_to_rgb(hex_color))

    def ratio_at(lv: float) -> float:
        h = rgb_to_hex(*oklab_to_rgb([lv, a, b]))
        return float(contrast_ratio(h, background)["ratio"])

    # Direction: away from the background's lightness (light bg -> darken,
    # dark bg -> lighten). Ties go to darkening (larger numeric ratio check).
    extreme = 0.0 if ratio_at(0.0) >= ratio_at(1.0) else 1.0
    lo_t, hi_t = 0.0, 1.0
    if ratio_at(L) >= target:
        # already compliant: no move needed
        return {"ok": True, "hex": rgb_to_hex(*oklab_to_rgb([L, a, b])),
                "ratio": round(ratio_at(L), 2),
                "note": "already meets the target; no lightness move needed"}
    if ratio_at(extreme) < target:
        return {"ok": False, "hex": rgb_to_hex(*oklab_to_rgb([extreme, a, b])),
                "ratio": round(ratio_at(extreme), 2),
                "note": (f"this hue cannot reach {target}:1 on {background} "
                         "— pick a different hue or accept the failing pair")}
    best_l = extreme
    for _ in range(steps):
        mid_t = (lo_t + hi_t) / 2
        mid_l = L + mid_t * (extreme - L)
        if ratio_at(mid_l) >= target:
            hi_t = mid_t
            best_l = mid_l
        else:
            lo_t = mid_t
        if hi_t - lo_t < 1e-3:
            break
    if ratio_at(best_l) < target:  # bisection never crossed: take the extreme
        best_l = extreme
    return {
        "ok": True,
        "hex": rgb_to_hex(*oklab_to_rgb([best_l, a, b])),
        "ratio": round(ratio_at(best_l), 2),
        "note": ("lightness moved along OKLab L (hue/chroma fixed) until the "
                 "WCAG target was met"),
    }
