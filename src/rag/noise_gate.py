import re
import statistics
import string
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ALLOWED = set(string.ascii_letters + string.digits + string.punctuation + " \n\t")
_TOKEN_RE = re.compile(r"\S+")

# patterns that often appear with OCR/typo corruption
_RE_MIXED_ALNUM = re.compile(r"([A-Za-z]\d|\d[A-Za-z])")
_RE_REPEAT_PUNCT = re.compile(r"([?!.,;:])\1{1,}")
_RE_WEIRD_SPACES = re.compile(r"[ \t]{3,}")


def _percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    xs2 = sorted(float(x) for x in xs)
    k = int(round((p / 100.0) * (len(xs2) - 1)))
    k = max(0, min(k, len(xs2) - 1))
    return float(xs2[k])


def noise_breakdown(text: str) -> Dict[str, float]:
    """
    Heuristic noise features for cheap OCR/typo gating.
    The important fix vs. the old version is that pattern counts are normalized
    by token count rather than added as flat constants.
    """
    if not text:
        return {
            "score": 0.0,
            "n_chars": 0.0,
            "n_tokens": 0.0,
            "non_ascii_count": 0.0,
            "disallowed_count": 0.0,
            "mixed_count": 0.0,
            "repeat_punct_count": 0.0,
            "weird_spaces_count": 0.0,
            "non_ascii_ratio": 0.0,
            "disallowed_ratio": 0.0,
            "mixed_ratio": 0.0,
            "repeat_punct_ratio": 0.0,
            "weird_spaces_ratio": 0.0,
        }

    n_chars = max(len(text), 1)
    n_tokens = max(len(_TOKEN_RE.findall(text)), 1)

    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    disallowed = sum(1 for ch in text if ch not in _ALLOWED)
    mixed = len(_RE_MIXED_ALNUM.findall(text))
    repeat_punct = len(_RE_REPEAT_PUNCT.findall(text))
    weird_spaces = len(_RE_WEIRD_SPACES.findall(text))

    non_ascii_ratio = non_ascii / n_chars
    disallowed_ratio = disallowed / n_chars
    mixed_ratio = mixed / n_tokens
    repeat_punct_ratio = repeat_punct / n_tokens
    weird_spaces_ratio = weird_spaces / n_tokens

    score = 0.0
    score += 2.0 * non_ascii_ratio
    score += 3.0 * disallowed_ratio
    score += 0.75 * mixed_ratio
    score += 0.50 * repeat_punct_ratio
    score += 0.50 * weird_spaces_ratio

    return {
        "score": float(score),
        "n_chars": float(n_chars),
        "n_tokens": float(n_tokens),
        "non_ascii_count": float(non_ascii),
        "disallowed_count": float(disallowed),
        "mixed_count": float(mixed),
        "repeat_punct_count": float(repeat_punct),
        "weird_spaces_count": float(weird_spaces),
        "non_ascii_ratio": float(non_ascii_ratio),
        "disallowed_ratio": float(disallowed_ratio),
        "mixed_ratio": float(mixed_ratio),
        "repeat_punct_ratio": float(repeat_punct_ratio),
        "weird_spaces_ratio": float(weird_spaces_ratio),
    }


def noise_score(text: str) -> float:
    return float(noise_breakdown(text)["score"])


def select_noisy_indices(
    texts: Sequence[str],
    threshold: float,
    max_chunks: int,
    percentile: Optional[float] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Returns selected indices plus rich debug info.

    Selection rule:
      score >= max(threshold, per-query percentile cutoff)
    This keeps the gate conservative:
      - threshold prevents denoising low-score clean chunks
      - percentile prevents selecting many chunks when several have similar scores
    """
    scored: List[Tuple[int, float, Dict[str, float]]] = []
    for i, t in enumerate(texts):
        info = noise_breakdown(t)
        scored.append((i, float(info["score"]), info))

    scored.sort(key=lambda x: x[1], reverse=True)
    raw_scores = [s for _, s, _ in scored]

    pct_cutoff = None
    if percentile is not None and raw_scores:
        pct_cutoff = _percentile(raw_scores, percentile)

    effective_threshold = max(float(threshold), float(pct_cutoff)) if pct_cutoff is not None else float(threshold)

    chosen = [
        i
        for i, s, _ in scored
        if s >= effective_threshold and s > 0.0
    ][: max_chunks]

    debug = {
        "score_min": float(min(raw_scores)) if raw_scores else 0.0,
        "score_median": float(statistics.median(raw_scores)) if raw_scores else 0.0,
        "score_max": float(max(raw_scores)) if raw_scores else 0.0,
        "score_p90": float(_percentile(raw_scores, 90)) if raw_scores else 0.0,
        "threshold": float(threshold),
        "percentile": float(percentile) if percentile is not None else None,
        "percentile_cutoff": float(pct_cutoff) if pct_cutoff is not None else None,
        "effective_threshold": float(effective_threshold),
        "selected_count": int(len(chosen)),
        "scored": [
            {
                "idx": int(i),
                "score": float(s),
                **info,
            }
            for i, s, info in scored
        ],
    }
    return chosen, debug