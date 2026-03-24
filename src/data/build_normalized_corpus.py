# src/data/normalize.py
import re
import unicodedata

_TRANSLATE = str.maketrans({
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201C": '"',  # left double quote
    "\u201D": '"',  # right double quote
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u00A0": " ",  # non-breaking space
    "\u200B": "",   # zero-width space
    "\uFEFF": "",   # BOM
})

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")

def normalize_text(text: str) -> str:
    """
    Safe normalization:
    - Unicode NFKC
    - normalize quotes/dashes
    - remove zero-width chars
    - collapse whitespace
    - collapse excessive newlines
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_TRANSLATE)
    t = _NL_RE.sub("\n\n", t)
    t = "\n".join(_WS_RE.sub(" ", line).strip() for line in t.split("\n"))
    return t.strip()