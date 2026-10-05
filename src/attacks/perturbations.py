"""
Adversarial text perturbations for benchmarking content-moderation classifiers.

Design principles
-----------------
1. Every attack is *readability preserving*. A perturbation that destroys human
   comprehension is not an evasion, it is vandalism, and it tells us nothing
   about real adversary behaviour. Attacks therefore avoid word-initial and
   word-final characters by default and are bounded by a perturbation rate.

2. Every attack is *budgeted*. `rate` is the fraction of eligible positions
   that get perturbed, so degradation can be reported as a curve rather than a
   single number.

3. Every attack is *seeded*. Results must be reproducible across runs.

4. Attacks can be *targeted*. Pass `word_budget` to restrict perturbation to
   specific word indices, which is what the query-guided greedy attacker in
   `search.py` uses once it has a saliency ranking.

Families
--------
visual      : looks near-identical to a human, different codepoints to a tokenizer
structural  : inserts characters/whitespace that humans skip over
linguistic  : transliteration and code-mixing (Hinglish), semantics preserved
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Attack",
    "AttackResult",
    "REGISTRY",
    "register",
    "get_attack",
    "list_attacks",
    "readability_report",
]


# --------------------------------------------------------------------------
# Character tables
# --------------------------------------------------------------------------

# Unicode confusables, split into two tiers.
#
# STRICT: glyph-identical to the Latin letter in common UI sans/serif fonts.
# A reader cannot tell the difference, so evasion via these characters is a
# pure tokenizer failure with no cost to the attacker. This tier carries the
# headline result, because it is the version no reviewer can wave away.
#
# LOOSE: recognisably different but still readable in context (Greek alpha for
# a, Cyrillic ghe for r). Real attackers use these, but a careful human reader
# notices them. Reported separately as an upper bound on attack strength.
#
# Letters with no faithful confusable (b, f, k, r, t, u, z) are absent from
# STRICT by design. Do not pad it; an incomplete strict table is the honest one.
CONFUSABLES_STRICT: Dict[str, Tuple[str, ...]] = {
    "a": ("\u0430",),              # CYRILLIC SMALL LETTER A
    "c": ("\u0441",),              # CYRILLIC SMALL LETTER ES
    "d": ("\u0501",),              # CYRILLIC SMALL LETTER KOMI DE
    "e": ("\u0435",),              # CYRILLIC SMALL LETTER IE
    "g": ("\u0261",),              # LATIN SMALL LETTER SCRIPT G
    "h": ("\u04bb",),              # CYRILLIC SMALL LETTER SHHA
    "i": ("\u0456",),              # CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    "j": ("\u0458",),              # CYRILLIC SMALL LETTER JE
    "l": ("\u04cf",),              # CYRILLIC SMALL LETTER PALOCHKA
    "m": ("\u217f",),              # SMALL ROMAN NUMERAL ONE THOUSAND
    "o": ("\u043e", "\u03bf"),     # CYRILLIC O, GREEK OMICRON
    "p": ("\u0440",),              # CYRILLIC SMALL LETTER ER
    "q": ("\u051b",),              # CYRILLIC SMALL LETTER QA
    "s": ("\u0455",),              # CYRILLIC SMALL LETTER DZE
    "v": ("\u0475",),              # CYRILLIC SMALL LETTER IZHITSA
    "w": ("\u051d",),              # CYRILLIC SMALL LETTER WE
    "x": ("\u0445",),              # CYRILLIC SMALL LETTER HA
    "y": ("\u0443",),              # CYRILLIC SMALL LETTER U
}

CONFUSABLES_LOOSE: Dict[str, Tuple[str, ...]] = {
    "a": ("\u03b1", "\u0251"),     # Greek alpha, Latin alpha
    "b": ("\u0184", "\u1e05"),
    "c": ("\u03f2",),              # lunate sigma
    "e": ("\u04bd", "\u212e"),
    "i": ("\u0131", "\u2170"),     # dotless i
    "k": ("\u043a", "\u03ba"),     # Cyrillic ka, Greek kappa
    "l": ("\u2113", "\u217c"),
    "n": ("\u0578", "\u1d0e"),     # Armenian vo, reversed small-cap N
    "r": ("\u0433", "\u1d26"),     # Cyrillic ghe
    "t": ("\u0442", "\u03c4"),     # Cyrillic te, Greek tau
    "u": ("\u057d", "\u03c5"),     # Armenian seh, Greek upsilon
    "v": ("\u03bd",),              # Greek nu
    "z": ("\u1d22", "\u0290"),
}

# Union, used by the readability guard to undo homoglyphs the way a human eye
# undoes them for free.
CONFUSABLES: Dict[str, Tuple[str, ...]] = {
    k: CONFUSABLES_STRICT.get(k, ()) + CONFUSABLES_LOOSE.get(k, ())
    for k in set(CONFUSABLES_STRICT) | set(CONFUSABLES_LOOSE)
}

# Reverse map: confusable codepoint -> the Latin letter it imitates.
_CONFUSABLE_REVERSE: Dict[str, str] = {
    ch: latin for latin, opts in CONFUSABLES.items() for ch in opts
}

# Mathematical alphanumeric symbols (U+1D400 block) plus enclosed and fullwidth
# forms. These render as normal-looking styled text but are entirely distinct
# codepoints, so subword tokenizers fall straight through to <unk> or byte
# fallback. Heavily used in the wild precisely because they survive copy-paste.
_MATH_STYLE_OFFSETS: Dict[str, Tuple[int, int]] = {
    # style          (uppercase base, lowercase base)
    "bold":          (0x1D400, 0x1D41A),
    "italic":        (0x1D434, 0x1D44E),
    "bold_italic":   (0x1D468, 0x1D482),
    "script":        (0x1D49C, 0x1D4B6),
    "fraktur":       (0x1D504, 0x1D51E),
    "double_struck": (0x1D538, 0x1D552),
    "sans":          (0x1D5A0, 0x1D5BA),
    "sans_bold":     (0x1D5D4, 0x1D5EE),
    "monospace":     (0x1D670, 0x1D68A),
}

_ENCLOSED_LOWER_BASE = 0x24D0   # circled a
_FULLWIDTH_LOWER_BASE = 0xFF41  # fullwidth a
_FULLWIDTH_UPPER_BASE = 0xFF21

# Zero-width and invisible formatting characters.
INVISIBLES: Tuple[str, ...] = (
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2060",  # word joiner
    "\u00ad",  # soft hyphen
    "\ufeff",  # zero width no-break space
)

# Combining diacritical marks (U+0300..U+036F). Applied sparsely so the text
# stays legible; dense application is the "Zalgo" effect and fails the guard.
COMBINING_MARKS: Tuple[str, ...] = tuple(chr(cp) for cp in range(0x0300, 0x0370))

LEET_MAP: Dict[str, Tuple[str, ...]] = {
    "a": ("4", "@"),
    "b": ("8",),
    "e": ("3",),
    "g": ("9", "6"),
    "i": ("1", "!"),
    "l": ("1", "|"),
    "o": ("0",),
    "s": ("5", "$"),
    "t": ("7", "+"),
    "z": ("2",),
}

PUNCT_FILLERS: Tuple[str, ...] = (".", "-", "_", "*", "'", "~")

_WORD_RE = re.compile(r"\w+", re.UNICODE)


# --------------------------------------------------------------------------
# Core types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AttackResult:
    """A perturbed string plus the bookkeeping needed for the readability guard."""

    original: str
    perturbed: str
    attack: str
    family: str
    rate: float
    n_edits: int

    @property
    def changed(self) -> bool:
        return self.original != self.perturbed


@dataclass
class Attack:
    """A named, budgeted, seeded perturbation."""

    name: str
    family: str
    fn: Callable[..., Tuple[str, int]]
    description: str = ""
    # Attacks that change token *count* (spacing, punctuation) are flagged
    # because they interact differently with subword tokenizers.
    splits_tokens: bool = False

    def __call__(
        self,
        text: str,
        rate: float = 0.3,
        seed: int = 0,
        word_budget: Optional[Sequence[int]] = None,
    ) -> AttackResult:
        rng = random.Random(seed)
        perturbed, n_edits = self.fn(text, rate=rate, rng=rng, word_budget=word_budget)
        return AttackResult(
            original=text,
            perturbed=perturbed,
            attack=self.name,
            family=self.family,
            rate=rate,
            n_edits=n_edits,
        )


REGISTRY: Dict[str, Attack] = {}


def register(
    name: str, family: str, description: str = "", splits_tokens: bool = False
) -> Callable:
    def deco(fn: Callable[..., Tuple[str, int]]) -> Callable:
        REGISTRY[name] = Attack(
            name=name,
            family=family,
            fn=fn,
            description=description,
            splits_tokens=splits_tokens,
        )
        return fn

    return deco


def get_attack(name: str) -> Attack:
    if name not in REGISTRY:
        raise KeyError(f"unknown attack {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def list_attacks(family: Optional[str] = None) -> List[str]:
    return sorted(
        n for n, a in REGISTRY.items() if family is None or a.family == family
    )


# --------------------------------------------------------------------------
# Position selection
# --------------------------------------------------------------------------

def _eligible_positions(
    text: str,
    rng: random.Random,
    rate: float,
    predicate: Callable[[str], bool],
    preserve_first_last: bool = True,
    word_budget: Optional[Sequence[int]] = None,
) -> List[int]:
    """Choose character indices to perturb.

    Word-initial and word-final characters are protected by default: they carry
    most of the visual word shape that human readers rely on, so leaving them
    intact keeps the text legible while still breaking subword tokenization.
    """
    words = list(_WORD_RE.finditer(text))
    if word_budget is not None:
        allowed = set(word_budget)
        words = [w for i, w in enumerate(words) if i in allowed]

    candidates: List[int] = []
    for w in words:
        start, end = w.start(), w.end()
        lo = start + 1 if (preserve_first_last and end - start > 2) else start
        hi = end - 1 if (preserve_first_last and end - start > 2) else end
        for i in range(lo, hi):
            if predicate(text[i]):
                candidates.append(i)

    if not candidates:
        return []
    k = max(1, round(len(candidates) * rate))
    k = min(k, len(candidates))
    return sorted(rng.sample(candidates, k))


def _is_letter(ch: str) -> bool:
    return ch.isalpha()


# --------------------------------------------------------------------------
# Visual family
# --------------------------------------------------------------------------

def _homoglyph(text, rate, rng, word_budget, table):
    idxs = _eligible_positions(
        text, rng, rate, lambda c: c.lower() in table, word_budget=word_budget
    )
    chars = list(text)
    n = 0
    for i in idxs:
        chars[i] = rng.choice(table[chars[i].lower()])
        n += 1
    return "".join(chars), n


@register(
    "homoglyph",
    "visual",
    "Glyph-identical Cyrillic/Latin substitutions only. Zero perceptible change.",
)
def homoglyph(text, rate, rng, word_budget=None):
    return _homoglyph(text, rate, rng, word_budget, CONFUSABLES_STRICT)


@register(
    "homoglyph_loose",
    "visual",
    "Adds recognisably-different confusables (Greek alpha, Cyrillic ghe). Upper bound.",
)
def homoglyph_loose(text, rate, rng, word_budget=None):
    return _homoglyph(text, rate, rng, word_budget, CONFUSABLES)


@register(
    "math_alnum",
    "visual",
    "Map letters into the Mathematical Alphanumeric Symbols block (bold/italic/script/etc).",
)
def math_alnum(text, rate, rng, word_budget=None):
    style = rng.choice(list(_MATH_STYLE_OFFSETS))
    up_base, lo_base = _MATH_STYLE_OFFSETS[style]
    idxs = _eligible_positions(
        text, rng, rate, lambda c: "a" <= c.lower() <= "z", word_budget=word_budget
    )
    chars = list(text)
    n = 0
    for i in idxs:
        ch = chars[i]
        base = lo_base if ch.islower() else up_base
        chars[i] = chr(base + (ord(ch.lower() if ch.islower() else ch.upper())
                               - (ord("a") if ch.islower() else ord("A"))))
        n += 1
    return "".join(chars), n


@register(
    "fullwidth",
    "visual",
    "Map ASCII letters to fullwidth forms (U+FF21..U+FF5A).",
)
def fullwidth(text, rate, rng, word_budget=None):
    idxs = _eligible_positions(
        text, rng, rate, lambda c: "a" <= c.lower() <= "z", word_budget=word_budget
    )
    chars = list(text)
    n = 0
    for i in idxs:
        ch = chars[i]
        base = _FULLWIDTH_LOWER_BASE if ch.islower() else _FULLWIDTH_UPPER_BASE
        ref = "a" if ch.islower() else "A"
        chars[i] = chr(base + (ord(ch) - ord(ref)))
        n += 1
    return "".join(chars), n


@register("leet", "visual", "Classic leetspeak digit and symbol substitution.")
def leet(text, rate, rng, word_budget=None):
    idxs = _eligible_positions(
        text, rng, rate, lambda c: c.lower() in LEET_MAP, word_budget=word_budget
    )
    chars = list(text)
    n = 0
    for i in idxs:
        chars[i] = rng.choice(LEET_MAP[chars[i].lower()])
        n += 1
    return "".join(chars), n


@register(
    "diacritic",
    "visual",
    "Attach a single combining mark to selected letters (sparse, not Zalgo).",
)
def diacritic(text, rate, rng, word_budget=None):
    idxs = _eligible_positions(text, rng, rate, _is_letter, word_budget=word_budget)
    out: List[str] = []
    marked = set(idxs)
    n = 0
    for i, ch in enumerate(text):
        out.append(ch)
        if i in marked:
            out.append(rng.choice(COMBINING_MARKS))
            n += 1
    return "".join(out), n


@register(
    "char_swap",
    "visual",
    "Transpose adjacent interior characters (DeepWordBug-style typo).",
)
def char_swap(text, rate, rng, word_budget=None):
    words = list(_WORD_RE.finditer(text))
    if word_budget is not None:
        allowed = set(word_budget)
        words = [w for i, w in enumerate(words) if i in allowed]
    words = [w for w in words if w.end() - w.start() >= 4]
    if not words:
        return text, 0
    k = max(1, round(len(words) * rate))
    chosen = rng.sample(words, min(k, len(words)))
    chars = list(text)
    n = 0
    for w in chosen:
        i = rng.randrange(w.start() + 1, w.end() - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        n += 1
    return "".join(chars), n


# --------------------------------------------------------------------------
# Structural family
# --------------------------------------------------------------------------

@register(
    "invisible",
    "structural",
    "Insert zero-width / soft-hyphen characters inside words.",
    splits_tokens=True,
)
def invisible(text, rate, rng, word_budget=None):
    idxs = _eligible_positions(text, rng, rate, _is_letter, word_budget=word_budget)
    marked = set(idxs)
    out: List[str] = []
    n = 0
    for i, ch in enumerate(text):
        out.append(ch)
        if i in marked:
            out.append(rng.choice(INVISIBLES))
            n += 1
    return "".join(out), n


@register(
    "intraword_space",
    "structural",
    "Insert spaces inside words, splitting them into fragments.",
    splits_tokens=True,
)
def intraword_space(text, rate, rng, word_budget=None):
    idxs = _eligible_positions(text, rng, rate, _is_letter, word_budget=word_budget)
    marked = set(idxs)
    out: List[str] = []
    n = 0
    for i, ch in enumerate(text):
        out.append(ch)
        if i in marked:
            out.append(" ")
            n += 1
    return "".join(out), n


@register(
    "punct_insert",
    "structural",
    "Insert punctuation fillers inside words (f.u.c.k style).",
    splits_tokens=True,
)
def punct_insert(text, rate, rng, word_budget=None):
    idxs = _eligible_positions(text, rng, rate, _is_letter, word_budget=word_budget)
    marked = set(idxs)
    out: List[str] = []
    n = 0
    for i, ch in enumerate(text):
        out.append(ch)
        if i in marked:
            out.append(rng.choice(PUNCT_FILLERS))
            n += 1
    return "".join(out), n


@register(
    "char_repeat",
    "structural",
    "Duplicate interior characters (looooser).",
    splits_tokens=True,
)
def char_repeat(text, rate, rng, word_budget=None):
    idxs = _eligible_positions(text, rng, rate, _is_letter, word_budget=word_budget)
    marked = set(idxs)
    out: List[str] = []
    n = 0
    for i, ch in enumerate(text):
        out.append(ch)
        if i in marked:
            out.append(ch * rng.randint(1, 2))
            n += 1
    return "".join(out), n


# --------------------------------------------------------------------------
# Composite
# --------------------------------------------------------------------------

def compose(*names: str) -> Attack:
    """Chain attacks, splitting the budget evenly so total edits stay comparable."""
    per = len(names)

    def fn(text, rate, rng, word_budget=None):
        total = 0
        cur = text
        for nm in names:
            atk = get_attack(nm)
            cur, k = atk.fn(cur, rate=rate / per, rng=rng, word_budget=None)
            total += k
        return cur, total

    return Attack(
        name="+".join(names),
        family="composite",
        fn=fn,
        description=f"Composite of {', '.join(names)}",
        splits_tokens=any(get_attack(n).splits_tokens for n in names),
    )


# --------------------------------------------------------------------------
# Readability guard
# --------------------------------------------------------------------------

def _strip_to_skeleton(s: str) -> str:
    """Approximate what a *human reader* recovers from perturbed text.

    Note the confusables reversal. A reader maps Cyrillic о to o for free, so a
    faithful model of human perception must too. Unicode normalisation cannot:
    NFKC/NFKD leave Cyrillic and Latin entirely unrelated, because they are
    genuinely different letters in different scripts.

    That gap is the thesis of the project. Visual attacks cost the reader
    nothing and cost the standard normalisation defence everything, which is
    why a confusables map has to be an explicit component of any moderation
    pipeline rather than something assumed to come free with NFKC.
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = "".join(c for c in s if c not in INVISIBLES)
    s = "".join(_CONFUSABLE_REVERSE.get(c, c) for c in s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def readability_report(result: AttackResult) -> Dict[str, float]:
    """Cheap automatic proxy for semantic preservation.

    `skeleton_distance` is the edit distance after undoing the perturbations a
    human visual system undoes for free. A homoglyph attack should score near
    zero here while still destroying model performance, which is exactly the
    result worth reporting. Hand-rate a 100-example sample to validate this
    proxy before trusting it in the paper.
    """
    orig_sk = _strip_to_skeleton(result.original)
    pert_sk = _strip_to_skeleton(result.perturbed)
    dist = _levenshtein(orig_sk, pert_sk)
    denom = max(len(orig_sk), 1)
    return {
        "skeleton_distance": dist,
        "skeleton_distance_norm": dist / denom,
        "raw_len_delta": len(result.perturbed) - len(result.original),
        "n_edits": result.n_edits,
    }


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    sample = "you are a complete idiot and everyone hates you"
    print(f"{'attack':<18} {'family':<11} {'skel':>5}  text")
    print("-" * 96)
    print(f"{'<clean>':<18} {'-':<11} {0.0:>5.2f}  {sample}")
    for name in list_attacks():
        atk = get_attack(name)
        res = atk(sample, rate=0.35, seed=7)
        rep = readability_report(res)
        print(
            f"{name:<18} {atk.family:<11} "
            f"{rep['skeleton_distance_norm']:>5.2f}  {res.perturbed}"
        )
    combo = compose("homoglyph", "invisible")
    res = combo(sample, rate=0.5, seed=7)
    rep = readability_report(res)
    print(
        f"{combo.name:<18} {combo.family:<11} "
        f"{rep['skeleton_distance_norm']:>5.2f}  {res.perturbed}"
    )
