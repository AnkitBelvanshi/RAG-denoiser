\
import random
import string
from dataclasses import dataclass
from typing import Dict

_QWERTY_NEIGHBORS: Dict[str, str] = {
    "a": "qwsxz",
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

_PRINTABLE_NO_WS = "".join(ch for ch in string.printable if not ch.isspace())


@dataclass
class NoiseConfig:
    seed: int = 42
    edits_per_100_chars: float = 2.5
    op_weights: Dict[str, float] = None
    preserve_whitespace: bool = True


def _choose_op(rng: random.Random, op_weights: Dict[str, float]) -> str:
    ops = list(op_weights.keys())
    w = [float(op_weights[o]) for o in ops]
    total = sum(w)
    if total <= 0:
        return ops[0]
    r = rng.random() * total
    acc = 0.0
    for op, wt in zip(ops, w):
        acc += wt
        if r <= acc:
            return op
    return ops[-1]


def _keyboard_typo(rng: random.Random, s: str, idx: int) -> str:
    ch = s[idx]
    lower = ch.lower()
    if lower in _QWERTY_NEIGHBORS:
        repl = rng.choice(_QWERTY_NEIGHBORS[lower])
        if ch.isupper():
            repl = repl.upper()
        return s[:idx] + repl + s[idx + 1:]
    repl = rng.choice(string.ascii_letters)
    return s[:idx] + repl + s[idx + 1:]


def _swap_adjacent(s: str, idx: int) -> str:
    if idx >= len(s) - 1:
        return s
    return s[:idx] + s[idx + 1] + s[idx] + s[idx + 2:]


def _delete_char(s: str, idx: int) -> str:
    return s[:idx] + s[idx + 1:]


def _insert_char(rng: random.Random, s: str, idx: int) -> str:
    ch = rng.choice(string.ascii_letters)
    return s[:idx] + ch + s[idx:]


def _random_replace(rng: random.Random, s: str, idx: int, preserve_whitespace: bool) -> str:
    if preserve_whitespace and s[idx].isspace():
        return s
    repl = rng.choice(_PRINTABLE_NO_WS) if preserve_whitespace else rng.choice(string.printable)
    return s[:idx] + repl + s[idx + 1:]


def perturb_text(text: str, cfg: NoiseConfig) -> str:
    if not text:
        return text

    rng = random.Random(cfg.seed)
    op_weights = cfg.op_weights or {
        "keyboard_typo": 0.35,
        "swap_adjacent": 0.20,
        "delete_char": 0.15,
        "insert_char": 0.15,
        "random_replace": 0.15,
    }

    n = len(text)
    expected = (cfg.edits_per_100_chars / 100.0) * n
    num_edits = max(0, int(round(expected + rng.random() - 0.5)))

    s = text
    for _ in range(num_edits):
        if not s:
            break
        idx = rng.randrange(0, len(s))

        if cfg.preserve_whitespace and s[idx].isspace():
            for _ in range(3):
                idx = rng.randrange(0, len(s))
                if not s[idx].isspace():
                    break

        op = _choose_op(rng, op_weights)

        if op == "keyboard_typo":
            s = _keyboard_typo(rng, s, idx)
        elif op == "swap_adjacent":
            s = _swap_adjacent(s, idx)
        elif op == "delete_char":
            s = _delete_char(s, idx)
        elif op == "insert_char":
            s = _insert_char(rng, s, idx)
        elif op == "random_replace":
            s = _random_replace(rng, s, idx, cfg.preserve_whitespace)
        else:
            s = _random_replace(rng, s, idx, cfg.preserve_whitespace)
    return s
