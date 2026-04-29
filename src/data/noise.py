import random
import string
from dataclasses import dataclass
from typing import Dict, Optional

KEYBOARD_NEIGHBORS = {
    "a": "qwsz",
    "b": "vghn",
    "c": "xdfv",
    "d": "serfcx",
    "e": "wsdr",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "i": "ujko",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tghu",
    "z": "asx",
}

DEFAULT_OP_WEIGHTS = {
    "swap": 0.4,
    "delete": 0.2,
    "insert": 0.2,
    "substitute": 0.2,
}


@dataclass
class NoiseConfig:
    seed: int = 42
    edits_per_100_chars: float = 1.0
    op_weights: Optional[Dict[str, float]] = None
    preserve_whitespace: bool = True


def _replace_with_keyboard_neighbor(ch: str, rng: random.Random) -> str:
    low = ch.lower()
    if low in KEYBOARD_NEIGHBORS:
        rep = rng.choice(KEYBOARD_NEIGHBORS[low])
        return rep.upper() if ch.isupper() else rep
    return ch


def _normalized_weights(op_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    weights = dict(DEFAULT_OP_WEIGHTS)
    if op_weights:
        for k, v in op_weights.items():
            if k in weights:
                weights[k] = float(v)

    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        return dict(DEFAULT_OP_WEIGHTS)

    return {k: max(v, 0.0) / total for k, v in weights.items()}


def perturb_text(text: str, cfg: NoiseConfig) -> str:
    """
    Apply lightweight character-level noise controlled by:
    - edits_per_100_chars: total expected edits per 100 chars
    - op_weights: relative distribution across swap/delete/insert/substitute
    - preserve_whitespace: avoid modifying whitespace characters
    """
    if not text:
        return text

    rng = random.Random(cfg.seed)
    weights = _normalized_weights(cfg.op_weights)

    total_edit_prob = max(0.0, float(cfg.edits_per_100_chars)) / 100.0
    p_swap = total_edit_prob * weights["swap"]
    p_delete = total_edit_prob * weights["delete"]
    p_insert = total_edit_prob * weights["insert"]
    p_substitute = total_edit_prob * weights["substitute"]

    chars = list(text)
    out = []
    i = 0

    while i < len(chars):
        ch = chars[i]

        if cfg.preserve_whitespace and ch.isspace():
            out.append(ch)
            i += 1
            continue

        # delete
        if rng.random() < p_delete:
            i += 1
            continue

        # swap adjacent
        if (
            i + 1 < len(chars)
            and rng.random() < p_swap
            and not chars[i + 1].isspace()
        ):
            out.append(chars[i + 1])
            out.append(chars[i])
            i += 2
            continue

        # substitute / typo
        if rng.random() < p_substitute and ch.isalpha():
            out.append(_replace_with_keyboard_neighbor(ch, rng))
        else:
            out.append(ch)

        # insert random lowercase character
        if rng.random() < p_insert:
            out.append(rng.choice(string.ascii_lowercase))

        i += 1

    return "".join(out)


def corrupt_text(
    text: str,
    rng: Optional[random.Random] = None,
    p_typo: float = 0.03,
    p_swap: float = 0.01,
    p_delete: float = 0.01,
    p_insert: float = 0.01,
) -> str:
    """
    Backward-compatible convenience wrapper.
    """
    if rng is None:
        rng = random.Random(0)

    cfg = NoiseConfig(
        seed=rng.randrange(0, 2**31 - 1),
        edits_per_100_chars=100.0 * (p_typo + p_swap + p_delete + p_insert),
        op_weights={
            "substitute": p_typo,
            "swap": p_swap,
            "delete": p_delete,
            "insert": p_insert,
        },
        preserve_whitespace=True,
    )
    return perturb_text(text, cfg)