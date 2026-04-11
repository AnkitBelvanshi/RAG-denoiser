import random
import string
from typing import Optional

# Simple character-level noise useful for denoiser training.
# Keep it lightweight and controllable.

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

def _replace_with_keyboard_neighbor(ch: str, rng: random.Random) -> str:
    low = ch.lower()
    if low in KEYBOARD_NEIGHBORS:
        rep = rng.choice(KEYBOARD_NEIGHBORS[low])
        return rep.upper() if ch.isupper() else rep
    return ch

def corrupt_text(
    text: str,
    rng: Optional[random.Random] = None,
    p_typo: float = 0.03,
    p_swap: float = 0.01,
    p_delete: float = 0.01,
    p_insert: float = 0.01,
) -> str:
    """
    Low-level character noise:
    - keyboard typos (replace char with neighbor)
    - swap adjacent
    - delete char
    - insert random char

    Keep probabilities small (typo-level).
    """
    if rng is None:
        rng = random.Random(0)

    if not text:
        return text

    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        ch = chars[i]

        # delete
        if rng.random() < p_delete and ch not in "\n":
            i += 1
            continue

        # swap adjacent
        if i + 1 < len(chars) and rng.random() < p_swap and chars[i] not in "\n" and chars[i+1] not in "\n":
            out.append(chars[i + 1])
            out.append(chars[i])
            i += 2
            continue

        # typo replace
        if rng.random() < p_typo and ch.isalpha():
            out.append(_replace_with_keyboard_neighbor(ch, rng))
        else:
            out.append(ch)

        # insert
        if rng.random() < p_insert and ch not in "\n":
            out.append(rng.choice(string.ascii_lowercase))

        i += 1

    return "".join(out)