## Module D1 — Exact self-similarity

This module is the **first safeguard** of department D.
It checks whether the mutant peptide is **identical** to a known human peptide.

This is the most basic and most critical safety check.

---

### Biological context (intuition with no background required)

The immune system is educated to **tolerate self-peptides**.

If a mutant peptide turns out to be identical
to a peptide normally produced by the organism:
- it will **not be recognized** as foreign,
- it could trigger an **autoimmune reaction** if used as a vaccine.

In both cases, it is a candidate to **eliminate**.

---

### Biological hypothesis (explicitly stated)

A peptide identical to a peptide from the normal human proteome
cannot be considered a neo-epitope.

This module therefore implements the following hypothesis:

> *"A mutant peptide identical to a known human peptide
> must be strongly penalized,
> because it will not be perceived as non-self
> by the immune system."*

---

### What this module models (and what it does not model)

#### Modeled
- exact comparison (string matching) between the mutant peptide
  and a provided corpus of human peptides.

#### Not modeled
- actual immunological tolerance,
- human peptides not included in the corpus,
- effects of partial similarity (see module D2).

The provided corpus is **incomplete by nature**.
The module detects the most obvious matches.

---

### Expected work (detailed logic)

#### 1. Load the human peptide corpus

- Load the provided file (e.g., `human_peptides_small.txt`).
- The file contains one peptide per line (plain text format).
- Loading must be:
  - done only once (at module initialization),
  - robust to the absence of the file.

---

#### 2. Compare the mutant peptide to the corpus

- For each candidate, check whether `candidate.peptide_mut`
  is **exactly present** in the corpus.
- The comparison is case-sensitive
  (all uppercase by convention).

---

#### 3. Return a strong penalty if exact match

- If the peptide is found in the corpus:
  return a **strong penalty** (very negative value).
- If the peptide is not found:
  return a neutral score (e.g., 0.0).

The penalty must be strong enough
so that this peptide is ranked last,
regardless of its other scores.

---

### Inputs used

- `candidate.peptide_mut`
- Reference file: `data/human_peptides_small.txt`

This module must **not** use:
- the WT peptide,
- the HLA,
- scores from other modules.

---

### Expected output (interpretation)

- **Neutral score (0.0)**
  -> peptide not found in the corpus, no risk detected.

- **Very negative score**
  -> peptide identical to a known human peptide,
  -> risk of tolerance or autoimmunity.

The score is binary in its logic
(match or no match),
but expressed as a float to remain compatible with the pipeline.

---

### Edge cases to handle

- Missing or empty corpus file
- Empty peptide
- Non-standard amino acids

In these cases:
- return a documented neutral score,
- never crash the pipeline.

If the corpus file is absent:
- document the chosen behavior (e.g., warning + neutral score),
- do not block the analysis.

---

### Common pitfalls (to absolutely avoid)

- Not loading the corpus at each call (performance)
- Forgetting to normalize case
- Using an approximate comparison (that is the role of D2)
- Returning a penalty that is too weak

---

### Limitations to state explicitly in the report

- The corpus is **incomplete**: it does not cover the entire human proteome.
- An exact match does not prove an autoimmune reaction.
- The absence of a match does not prove safety.
- Very similar peptides (1 AA difference) are not detected here.

---

### What this module does NOT claim to do

- It does not predict autoimmunity.
- It does not model thymic tolerance.
- It does not replace a clinical safety analysis.

It provides a **basic filter**,
necessary but not sufficient.

---

### Critical question you must be able to answer

> *"Why is an exact match with the self
> a sufficient reason to strongly penalize a candidate,
> and what are the limitations of this approach
> (false negatives, corpus incompleteness)?"*

If you can clearly answer this question,
then your module fulfills its role as a safeguard.
