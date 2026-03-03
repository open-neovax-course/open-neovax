## Module D2 — Approximate self-similarity (optional)

This module **complements** module D1
by detecting peptides that are not identical to the self,
but are **too close** to it.

It is an optional but recommended module,
which refines the safety filter.

---

### Biological context (intuition with no background required)

Module D1 detects **exact** matches.

But a peptide that differs from the self by a single amino acid
is often **almost as problematic**:
- it may be tolerated by the immune system,
- it has a low probability of being recognized as truly foreign.

Conversely, a peptide very different from the self
is more likely to be perceived as novel.

---

### Biological hypothesis (explicitly stated)

A mutant peptide very similar to the self
(differing by only 1 or 2 amino acids)
has a reduced probability of being immunogenic.

This module therefore implements the following hypothesis:

> *"Mutant peptides very close to the self
> (low Hamming distance relative to the human corpus)
> are penalized gradually,
> as a complement to the exact filter of module D1."*

---

### What this module models (and what it does not model)

#### Modeled (proxy)
- approximate similarity between the mutant peptide
  and the human peptide corpus,
- via **Hamming distance**.

#### Not modeled
- actual immunological tolerance,
- structural effects of substitutions,
- the functional difference between conservative and radical mutations.

---

### Hamming distance concept (reminder)

The **Hamming distance** between two strings of the same length
is the **number of positions** where the characters differ.

Examples (for 9-mers):
- `LLFGYPRLV` vs `LLFGYPRLV` -> distance = 0 (identical)
- `LLFGYPRLV` vs `LLFGYPRLA` -> distance = 1
- `LLFGYPRLV` vs `AAFGYPRLA` -> distance = 2

---

### Expected work (detailed logic)

#### 1. Load the human peptide corpus

- Reuse the same file as D1 (`human_peptides_small.txt`).
- Load only once at module startup.
- Keep only peptides of the same length as the candidate.

---

#### 2. Compute the minimum distance to the self

- For each peptide in the corpus (of the same length):
  - compute the Hamming distance with `candidate.peptide_mut`.
- Retain the **minimum distance** found.

---

#### 3. Apply a gradual penalty

Accepted approaches:

- distance = 0 -> maximum penalty (but this is the role of D1),
- distance = 1 -> strong penalty,
- distance = 2 -> moderate penalty,
- distance >= 3 -> no penalty (or zero penalty).

The penalty scale must be:
- documented,
- consistent,
- biologically justified (even simply).

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
  -> peptide sufficiently different from the self.

- **Negative score (gradual)**
  -> peptide too similar to the self,
  -> penalty proportional to proximity.

The score must be directly interpretable
in terms of distance to the self.

---

### Edge cases to handle

- Missing or empty corpus
- Empty peptide
- No peptide of the same length in the corpus
- Non-standard amino acids

In these cases:
- return a documented neutral score,
- never crash the pipeline.

---

### Performance considerations

This module compares the candidate peptide
to **every peptide in the corpus**.
For a reasonably sized corpus (a few thousand),
this is fast.

If the corpus is larger:
- consider preliminary filtering (by length),
- document the complexity.

---

### Common pitfalls (to absolutely avoid)

- Comparing peptides of different lengths
- Not filtering the corpus by length
- Applying the same penalty as D1 (this module must be gradual)
- Loading the corpus at each call

---

### Limitations to state explicitly in the report

- Hamming distance does not capture functional similarity.
- Two physically very different AAs count the same
  as two similar AAs (L -> I vs G -> W).
- The corpus is incomplete.
- The penalty threshold is a heuristic choice.

---

### What this module does NOT claim to do

- It does not predict immunogenicity.
- It does not model actual tolerance.
- It does not replace D1 (exact match).

It provides a **gradual signal of proximity to the self**,
complementary to the binary filter of D1.

---

### Critical question you must be able to answer

> *"Why is a peptide at Hamming distance 1 from the self
> penalized in your model,
> and what are the limitations of this similarity measure?"*

If you can clearly answer this question,
then your module correctly captures
the intuition of proximity to the self.
