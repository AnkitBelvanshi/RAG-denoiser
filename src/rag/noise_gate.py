# src/rag/noise_gate.py
import re
import string

_ALLOWED = set(string.ascii_letters + string.digits + string.punctuation + " \n\t")

# patterns that often appear with OCR/typo corruption
_RE_MIXED_ALNUM = re.compile(r"([A-Za-z]\d|\d[A-Za-z])")
_RE_REPEAT_PUNCT = re.compile(r"([?!.,;:])\1{1,}")
_RE_WEIRD_SPACES = re.compile(r"[ \t]{3,}")


def noise_score(text: str) -> float:
    """
    Heuristic noise score. Higher = more likely corrupted.
    Designed to be cheap and to catch OCR/typo artifacts without heavy models.
    """
    if not text:
        return 0.0

    n = max(len(text), 1)

    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    disallowed = sum(1 for ch in text if ch not in _ALLOWED)

    mixed = len(_RE_MIXED_ALNUM.findall(text))
    repeat_punct = len(_RE_REPEAT_PUNCT.findall(text))
    weird_spaces = len(_RE_WEIRD_SPACES.findall(text))

    # Normalize by length for char-based terms, keep pattern counts mild.
    score = 0.0
    score += 2.0 * (non_ascii / n)
    score += 3.0 * (disallowed / n)
    score += 0.02 * mixed
    score += 0.02 * repeat_punct
    score += 0.02 * weird_spaces

    return float(score)


def select_noisy_indices(texts, threshold: float, max_chunks: int):
    """
    Returns indices to denoise (sorted by score desc), limited by max_chunks.
    """
    scored = [(i, noise_score(t)) for i, t in enumerate(texts)]
    scored.sort(key=lambda x: x[1], reverse=True)

    chosen = [i for (i, s) in scored if s >= threshold]
    return chosen[: max_chunks], scored