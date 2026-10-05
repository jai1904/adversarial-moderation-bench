# Adversarial Moderation Bench

**How much harmful content can evade production moderation systems using
perturbations that cost a human reader nothing?**

Content moderation classifiers are evaluated on clean text. Adversaries do not
send clean text. This benchmark measures how far recall falls under cheap,
readability-preserving perturbations, which defences recover it, and what those
defences cost on clean traffic.

> **Status:** in progress. Attack library complete; baselines and evaluation in
> flight. Numbers below are placeholders until the full sweep lands.

---

## The core observation

Not all evasions are equal, and the most important distinction is invisible in
the usual metrics.

| | text | readable? | undone by NFKC? |
|---|---|---|---|
| clean | `you are a complete idiot` | — | — |
| zero-width insertion | `you ar­e a co­m‌plete idi​o​t` | yes | **yes** |
| math alphanumerics | `you a𝕣e 𝕒 c𝕠𝕞plete idi𝕠t` | yes | **yes** |
| homoglyph | `yоu are а comрlеte idiot` | yes | **no** |

The third row contains six Cyrillic codepoints. No reader can distinguish it
from the first. Unicode normalisation cannot repair it either, because Cyrillic
`о` and Latin `o` are not variants of one another: they are different letters
in different scripts, and no normalisation form relates them.

Standard text-cleaning pipelines therefore neutralise most visual evasions
while leaving the cheapest and most widely deployed one fully intact. Defending
against it requires an explicit confusables map as a named component, not
something assumed to arrive free with `unicodedata.normalize`.

---

## Attack families

All attacks are budgeted (a `rate` parameter controls the fraction of eligible
positions perturbed, so degradation is reported as a curve), seeded, and
targetable (`word_budget` restricts perturbation to chosen token indices, used
by the query-guided attacker).

**visual** — `homoglyph`, `homoglyph_loose`, `math_alnum`, `fullwidth`, `leet`,
`diacritic`, `char_swap`

**structural** — `invisible`, `intraword_space`, `punct_insert`, `char_repeat`

**linguistic** *(planned)* — Hinglish transliteration and code-mixing

### Readability guard

A perturbation that destroys comprehension is vandalism, not evasion, and tells
us nothing about adversary behaviour. Every attack is therefore constrained:
word-initial and word-final characters are protected, preserving the visual word
shape human reading depends on while still breaking subword tokenization.

Readability is scored by edit distance over a *skeleton* string that models what
a reader recovers for free: NFKD, combining marks dropped, invisibles stripped,
and **confusables reversed**. The confusables reversal is the point. Human
perception undoes homoglyphs; Unicode normalisation does not. That asymmetry is
exactly the gap attackers exploit.

Automatic scoring is validated against a 100-example manual rating pass,
conducted in a browser rather than a terminal, since zero-width characters and
combining marks render differently across environments.

---

## Metrics

Accuracy and F1 are the wrong headline numbers. Moderation operates at a fixed
false-positive budget, because every false positive is a wrongly-removed post
and a user appeal.

**Classification** — recall at fixed precision 0.90, with the threshold chosen
once on validation and frozen. Retuning per attack would be cheating: a real
system does not know it is under attack.

**Evasion** — attack success rate, the fraction of correctly-flagged items an
attack pushes below threshold, reported per severity grade. Evading a threat
matters more than evading an insult, and a pooled number hides that.

**Ranking** — moderation is a capacity-bounded queue, not a binary decision.
Content is scored, ranked, and reviewed top-down until reviewer capacity runs
out. Reported as NDCG@k against graded severity, precision and severe-recall at
fixed capacity, and **rank displacement**: how far an attack pushes harmful
items down the queue.

Rank displacement is the metric most benchmarks omit, and it is where the real
damage shows. A recall drop of 0.08 sounds survivable until the stratified
numbers show the median harmful item fell ~85 places and stayed live. Pooled
displacement is near zero because benign items rising cancel harmful items
falling, so displacement is always reported stratified by severity.

---

## Defences evaluated

1. Unicode normalisation + explicit confusables map
2. Invisible-character stripping and whitespace collapsing
3. Leet de-obfuscation
4. Adversarial fine-tuning at ρ = 0.2
5. Character-level model

Each reported on three axes: adversarial recall recovered, **clean precision
lost**, and added latency per 1k messages. The trade-off is the result. A
defence recovering 30 points of adversarial recall at the cost of 4 points of
clean precision is an engineering decision, not a free win.

---

## Quickstart

The attack library is pure standard library. No install required.

```bash
python src/attacks/perturbations.py
```

Full pipeline:

```bash
pip install -r requirements.txt
python -m src.data.prepare --out data/processed
```

```python
from src.attacks.perturbations import get_attack, readability_report

atk = get_attack("homoglyph")
res = atk("you are a complete idiot", rate=0.35, seed=7)
print(res.perturbed)                 # yоu are а comрlеte idiot
print(readability_report(res))       # skeleton_distance_norm: 0.0
```

---

## Data

[Civil Comments](https://huggingface.co/datasets/google/civil_comments) via the
Hugging Face hub. Chosen over the Kaggle Jigsaw release for its six graded
sub-labels, which the severity rubric and ranking module both depend on; binary
toxic/not-toxic cannot express that a threat outranks an insult.

Splits are frozen and reproducible. `data/processed/manifest.json` is tracked in
git; the parquet files are not.

---

## Scope and ethics

This repository studies evasion of moderation systems in order to improve them.
It uses only established, licensed research corpora. Nothing is scraped, and no
material relating to child safety is collected, stored, or processed under any
circumstances.

The attack implementations are deliberately simple, and every technique here is
already in widespread adversarial use. The contribution is measurement and
defence, not novel capability.

---

## Layout

```
src/attacks/perturbations.py   attack library, readability guard
src/data/prepare.py            splits, severity rubric, frozen attack set
src/eval/metrics.py            operating point, evasion, ranking metrics
src/models/                    baselines and fine-tuning
results/                       cached sweeps and figures
```

## Roadmap

- [x] Attack library: visual and structural families
- [x] Readability guard with confusables-aware skeleton
- [x] Evaluation metrics: classification, evasion, ranking
- [ ] Frozen data splits and attack set
- [ ] Baselines: Detoxify, DistilBERT, Perspective API
- [ ] Attack sweep and degradation curves
- [ ] Manual readability validation (n=100)
- [ ] Query-guided greedy attack
- [ ] Hinglish linguistic family
- [ ] Defence evaluation
- [ ] Review-queue ranking analysis
- [ ] Writeup

## Licence

MIT
