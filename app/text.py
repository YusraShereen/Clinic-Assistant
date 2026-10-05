"""Text utilities shared by the API and tests (no torch dependency)."""
import re

# The NLU was trained on whitespace-tokenised text without punctuation, so serving must
# normalise the same way. ':' is kept inside times such as 4:30.
_PUNCT = re.compile(r"[?!.,،؟۔;\"'()\[\]{}]")
_COLON = re.compile(r"(?<!\d):|:(?!\d)")
_ARABIC = re.compile(r"[\u0600-\u06FF]")

_ROMAN_UR = {
    "kya", "hai", "hain", "mujhe", "mera", "meri", "mere", "aap", "ke", "ki", "ka", "ko", "se", "mein",
    "nahi", "kal", "aaj", "parso", "chahiye", "karna", "karein", "kar", "dein", "dena", "hoon", "sakta",
    "sakti", "bhai", "salam", "shukriya", "kitni", "kitna", "kitne", "kab", "kahan", "koi", "ab", "baje",
    "subah", "shaam", "dopahar", "raat", "thi", "tha", "wala", "wali", "liye", "saath", "par", "pe", "din",
    "bohat", "bata", "batao", "bataein", "milna", "dikhana", "hota", "hote", "hoga", "karen", "kijiye",
}
_ENGLISH = {
    "the", "is", "are", "my", "i", "me", "you", "can", "want", "need", "with", "for", "to", "what", "how",
    "do", "does", "please", "a", "an", "of", "on", "in", "at", "it", "this", "that", "have", "has", "will",
    "would", "could", "much", "much", "when", "where", "which", "who", "your", "we", "am",
}


def normalize(text: str):
    t = _PUNCT.sub(" ", text or "")
    t = _COLON.sub(" ", t)
    return t.split()


def detect_lang(text: str):
    """-> (lang, confident). lang in {'en','ur','rur'}. Script decides Urdu; Latin text is
    split between English and Roman Urdu by function-word counts."""
    if _ARABIC.search(text or ""):
        return "ur", True
    toks = [w.lower() for w in normalize(text)]
    ur = sum(w in _ROMAN_UR for w in toks)
    en = sum(w in _ENGLISH for w in toks)
    if ur == 0 and en == 0:
        return "en", False
    if ur > en:
        return "rur", True
    if en > ur:
        return "en", True
    return "rur", False


MERGE_TYPES = {"DOCTOR", "DEPARTMENT", "DATE", "TIME"}   # multi-word entities; adjacent pieces are one entity


def decode_entities(tokens, tags, merge_adjacent=True):
    """BIO tags -> {TYPE: [text, ...]}.
    - An I- tag without an open span of the same type starts a new span (model quirk).
    - merge_adjacent: a B-X directly after a span of the same type X (X in MERGE_TYPES) continues it.
      The model sometimes tags the 2nd word of 'next week' as B-DATE; two separate dates are never
      written with no word between them, so merging is safe for these types."""
    out, cur = {}, None
    for i, t in enumerate(list(tags) + ["O"]):
        cont = (t != "O" and cur is not None and cur[0] == t[2:]
                and (t.startswith("I-") or (merge_adjacent and t[2:] in MERGE_TYPES)))
        if cont:
            continue
        if cur is not None:
            out.setdefault(cur[0], []).append(" ".join(tokens[cur[1]:i]))
            cur = None
        if t != "O":
            cur = (t[2:], i)
    return out
