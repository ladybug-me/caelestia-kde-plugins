"""genius.creative — color science and small generative ideation.

Color (the serious half — this is the math matugen gestures at):

  * sRGB <-> OKLab <-> OKLch (Björn Ottosson's perceptual model, exact
    constants, pure Python)
  * perceptual distance (dE-OK) between two colors
  * WCAG 2.x contrast ratio with AA/AAA verdicts for normal/large text
  * harmony palettes generated in OKLch (complementary, analogous,
    triadic, tetradic, split-complementary), hex output, seeded
  * auto-accent: derive a pleasing accent from an arbitrary base color
    by lightness rotation — the "what accent goes with my wallpaper"
    question answered with geometry, not vibes

Ideation (the playful half):

  * bisociation: forced cross-product of two word lists with similarity
    filtering (the classic creativity technique)
  * taglines from template grammars seeded by your domain words
  * time-boxed ideas: N distinct ideas in under a second, ranked by
    novelty against a vocabulary you supply

Everything is a proposal — palettes are printed as hex strings with the
reasoning, never written into any config.
"""
from __future__ import annotations

import math
import random
import re
from typing import Any, Dict, Optional, Sequence, Tuple

__all__ = [
    "hex_to_rgb", "rgb_to_hex", "rgb_to_oklab", "oklab_to_rgb",
    "oklab_to_oklch", "oklch_to_oklab", "hex_to_oklch", "oklch_to_hex",
    "color_distance", "contrast_ratio", "palette", "auto_accent",
    "bisociate", "tagline", "idea_sprint",
]


# ---------------------------------------------------------------------------
# Color spaces
# ---------------------------------------------------------------------------

def hex_to_rgb(hexstr: str) -> Tuple[float, float, float]:
    h = hexstr.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        raise ValueError(f"bad hex color {hexstr!r}")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(r: float, g: float, b: float) -> str:
    def byte(v: float) -> int:
        return max(0, min(255, round(v * 255)))
    return "#{:02x}{:02x}{:02x}".format(byte(r), byte(g), byte(b))


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    if c <= 0:
        return 0.0
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def rgb_to_oklab(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(c) for c in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def oklab_to_rgb(lab: Tuple[float, float, float]) -> Tuple[float, float, float]:
    L, a, b = lab
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b2 = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return tuple(_linear_to_srgb(max(0.0, min(1.0, v))) for v in (r, g, b2))  # type: ignore[return-value]


def oklab_to_oklch(lab: Tuple[float, float, float]) -> Tuple[float, float, float]:
    L, a, b = lab
    C = math.sqrt(a * a + b * b)
    H = math.degrees(math.atan2(b, a)) % 360
    return L, C, H


def oklch_to_oklab(l: float, c: float, h_deg: float) -> Tuple[float, float, float]:
    h = math.radians(h_deg)
    return l, c * math.cos(h), c * math.sin(h)


def hex_to_oklch(hexstr: str) -> Tuple[float, float, float]:
    return oklab_to_oklch(rgb_to_oklab(hex_to_rgb(hexstr)))


def oklch_to_hex(l: float, c: float, h: float) -> str:
    return rgb_to_hex(*oklab_to_rgb(oklch_to_oklab(l, c, h)))


def color_distance(hex_a: str, hex_b: str) -> Dict[str, Any]:
    """dE-OK: Euclidean distance in OKLab (1.0 ~ one just-noticeable step)."""
    a = rgb_to_oklab(hex_to_rgb(hex_a))
    b = rgb_to_oklab(hex_to_rgb(hex_b))
    de = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    verdict = ("indistinguishable" if de < 0.01 else
               "close" if de < 0.05 else
               "noticeably different" if de < 0.11 else
               "clearly different" if de < 0.22 else "opposite ends")
    return {"delta_e_ok": round(de, 4), "verdict": verdict}


def _relative_luminance(rgb: Tuple[float, float, float]) -> float:
    lin = [_srgb_to_linear(c) for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast_ratio(hex_a: str, hex_b: str) -> Dict[str, Any]:
    la, lb = _relative_luminance(hex_to_rgb(hex_a)), _relative_luminance(hex_to_rgb(hex_b))
    hi, lo = max(la, lb), min(la, lb)
    ratio = (hi + 0.05) / (lo + 0.05)
    return {"ratio": round(ratio, 3),
            "aa_normal_text": ratio >= 4.5, "aa_large_text": ratio >= 3.0,
            "aaa_normal_text": ratio >= 7.0, "aaa_large_text": ratio >= 4.5,
            "verdict": ("AAA normal" if ratio >= 7 else
                        "AA normal / AAA large" if ratio >= 4.5 else
                        "AA large only" if ratio >= 3 else
                        "fails AA — decoration only")}


_HARMONIES = {
    "complementary": [0, 180],
    "analogous": [0, 30, -30],
    "triadic": [0, 120, 240],
    "tetradic": [0, 90, 180, 270],
    "split_complementary": [0, 150, 210],
}


def palette(base_hex: str, harmony: str = "analogous", n: int = 5,
            seed: int = 7) -> Dict[str, Any]:
    """Generate a palette around a base color in OKLch (perceptual)."""
    if harmony not in _HARMONIES:
        raise ValueError(f"unknown harmony {harmony!r} "
                         f"({', '.join(_HARMONIES)})")
    L, C, H = hex_to_oklch(base_hex)
    rng = random.Random(seed)
    hues = list(_HARMONIES[harmony])
    while len(hues) < n:
        hues.append(rng.choice([hues[-1] + 25, hues[-1] - 25]) % 360)
    swatches = []
    for i in range(n):
        l = max(0.25, min(0.95, L + rng.uniform(-0.12, 0.12) if i else L))
        c = max(0.02, min(0.32, C * (1 - 0.18 * abs(i)) if i else C))
        h = (H + hues[i]) % 360
        hexv = oklch_to_hex(l, c, h)
        swatches.append({
            "hex": hexv, "oklch": [round(l, 4), round(c, 4), round(h, 1)],
            "role": "base" if i == 0 else f"swatch {i}",
            "contrast_on_white": contrast_ratio(hexv, "#ffffff")["ratio"],
            "contrast_on_black": contrast_ratio(hexv, "#000000")["ratio"],
        })
    return {"base": base_hex, "harmony": harmony,
            "base_oklch": [round(L, 4), round(C, 4), round(H, 1)],
            "swatches": swatches,
            "note": "proposal only — nothing here writes to a theme"}


def auto_accent(base_hex: str, mode: str = "auto") -> Dict[str, Any]:
    """Derive an accent from a base: darken-then-boost or hue-shift."""
    L, C, H = hex_to_oklch(base_hex)
    if mode == "auto":
        mode = "boost" if L > 0.6 else "brighten"
    if mode == "boost":
        target_l, target_c = max(0.45, L - 0.22), min(0.31, C * 1.6 + 0.04)
    elif mode == "brighten":
        target_l, target_c = min(0.78, L + 0.18), min(0.31, C * 1.35 + 0.02)
    elif mode == "hue_shift":
        target_l, target_c, H = L, min(0.31, C * 1.2 + 0.03), (H + 36) % 360
    else:
        raise ValueError("mode must be auto|boost|brighten|hue_shift")
    accent = oklch_to_hex(target_l, target_c, H)
    return {"base": base_hex, "mode": mode,
            "accent": accent, "accent_oklch": [round(target_l, 4), round(target_c, 4), round(H, 1)],
            "delta_e_ok": color_distance(base_hex, accent)["delta_e_ok"],
            "contrast": contrast_ratio(accent, "#ffffff"),
            "note": "accent geometry: keep hue, fix lightness for contrast, "
                    "raise chroma for presence"}


# ---------------------------------------------------------------------------
# Ideation
# ---------------------------------------------------------------------------

def bisociate(list_a: Sequence[str], list_b: Sequence[str],
              n: int = 8, seed: int = 3) -> Dict[str, Any]:
    """Forced associations between two domains (Koestler's bisociation)."""
    if not list_a or not list_b:
        raise ValueError("two non-empty word lists required")
    rng = random.Random(seed)
    pairs = []
    for _ in range(n * 4):
        a, b = rng.choice(list_a), rng.choice(list_b)
        if not a or not b or a.lower() == b.lower():
            continue
        novelty = 1.0 - _word_similarity(a, b)
        pairs.append({"a": a, "b": b,
                     "combination": f"{a} {b}",
                     "novelty": round(novelty, 3)})
    pairs.sort(key=lambda p: -p["novelty"])
    uniq, seen = [], set()
    for p in pairs:
        if p["combination"].lower() in seen:
            continue
        seen.add(p["combination"].lower())
        uniq.append(p)
    return {"technique": "bisociation (forced association)",
            "pairs": uniq[:n],
            "note": "the point is the collision, not the polish"}


def _word_similarity(a: str, b: str) -> float:
    sa = set(a.lower()) & set(b.lower())
    su = set(a.lower()) | set(b.lower())
    return len(sa) / len(su) if su else 0.0


def tagline(subject: str, tone: str = "confident", seed: int = 11) -> Dict[str, Any]:
    """Template-grammar taglines seeded by the subject's own words."""
    rng = random.Random(seed)
    words = [w for w in re.findall(r"[a-zA-Z]+", subject) if len(w) > 3]
    if not words:
        words = ["your", "thing"]
    key = rng.choice(words[:4])
    templates = {
        "confident": [
            "{key}, but {adj}",
            "the {adj} way to {key}",
            "{key} that {verb}",
        ],
        "playful": [
            "hold my {key}",
            "{key} goes brrr",
            "more {adj} than your {key}",
        ],
        "minimal": ["{key}.", "{adj} {key}.", "just {key}"],
    }[tone if tone in ("confident", "playful", "minimal") else "confident"]
    adjs = ["fast", "quiet", "relentless", "unreasonable", "elegant", "sharp"]
    verbs = ["ships", "learns", "persists", "compounds", "disappears"]
    out = []
    for t in templates:
        line = t.format(key=key, adj=rng.choice(adjs), verb=rng.choice(verbs))
        out.append(line)
    return {"subject": subject, "tone": tone, "taglines": out}


def idea_sprint(theme: str, vocabulary: Optional[Sequence[str]] = None,
                n: int = 6, seed: int = 5) -> Dict[str, Any]:
    """Time-boxed divergent ideation: forced pairs + template mashups."""
    vocab = [v for v in (vocabulary or []) if v] or [
        "second brain", "ledger", "mirror", "signals", "calibration",
        "memory decay", "bandit", "proposal", "evidence", "rhythm"]
    theme_words = [w for w in re.findall(r"[a-zA-Z]+", theme) if len(w) > 3] or ["idea"]
    rng = random.Random(seed)
    ideas = []
    for i in range(n):
        a = rng.choice(theme_words)
        b = rng.choice(vocab)
        angle = rng.choice(["reverse it", "make it daily", "make it tiny",
                            "remove the user", "make it propose only",
                            "make it honest about uncertainty"])
        ideas.append({"idea": f"{a} x {b}",
                      "angle": angle,
                      "one_liner": f"what if {b} handled {a} — {angle}?"})
    return {"theme": theme, "ideas": ideas,
            "note": "divergence first; judgment is a later, calmer step"}
