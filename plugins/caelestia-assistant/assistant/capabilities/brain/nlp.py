"""Tokenisation and shingling shared by the learners."""
import re

WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    "a an and are as at be by for from in is it of on or that the this to "
    "was with i my me we you your our so if not but be have has do".split()
)


def words(text):
    return WORD.findall(text.lower())


def tokens(text):
    return [w for w in words(text) if w not in STOP and len(w) > 1]


def shingles(text, k=3):
    w = words(text)
    if len(w) < k:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}
