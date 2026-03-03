## Module C1 — Anchor position P2 (PSSM HLA-I)

This module is the **first pillar** of department C.
It models one of the most robust and best-established constraints
of HLA class I presentation:
**anchoring at position 2 (P2)**.

For many HLA-I alleles,
position P2 is **absolutely critical** for binding.

---

### Biological context (intuition without required immuno background)

The HLA-I molecule has a **binding groove** that accommodates the peptide.

This groove:
- imposes geometric constraints,
- favors certain amino acids at specific positions,
- strongly penalizes others.

For 9-amino-acid peptides:
- **position 2 (P2)** is one of the main anchor points,
- if the amino acid at P2 is incompatible,
  the peptide **does not bind**, regardless of the rest of the sequence.

👉 You can think of P2 as an **anchor nail**:
if it does not fit into the pocket, the entire peptide is rejected.

---

### Biological hypothesis (explicitly stated)

Known HLA-I ligands show strong preferences
for certain amino acids at position 2,
specific to each HLA allele.

This module therefore implements the following hypothesis:

> *"A peptide whose amino acid at position 2
> matches the preferences of the considered HLA allele
> is much more likely to bind to the HLA."*

---

### Why use a PSSM?

A **PSSM (Position-Specific Scoring Matrix)** is a table
that associates:
- a peptide position,
- an amino acid,
- a score reflecting its frequency or enrichment
  among known ligands.

For this module:
- the PSSM is **provided**,
- it corresponds to a specific HLA allele (e.g., HLA-A*02:01),
- it encodes experimentally observed preferences.

👉 The PSSM is:
- explicit,
- interpretable,
- historically used before modern ML models.

---

### Expected work (detailed logic)

#### 1. Load the provided PSSM matrix

- Load the provided CSV file (e.g., `hla_pssm_A0201.csv`).
- The matrix associates:
  - one row = one amino acid,
  - one column = one position (1 to 9),
  - one value = preference score.

Loading must be:
- done only once (at module loading if possible),
- robust to format errors.

---

#### 2. Extract the amino acid at position 2

- Extract the 2nd character of `candidate.peptide_mut`.
- Verify that:
  - the peptide length is sufficient,
  - the amino acid is valid.

---

#### 3. Return the corresponding score

- Read from the matrix the value corresponding to:
  - the observed amino acid,
  - column P2.
- Return this score as-is (or normalized if documented).

⚠️ The module must **not**:
- sum all positions,
- combine with other anchors,
- modify the scale without justification.

---

### Inputs used

- `candidate.peptide_mut`

This module must **not** use:
- the WT peptide,
- the mutation position,
- other modules,
- the global score.

---

### Expected output (interpretation)

- **High / slightly negative score**
  → favorable amino acid at position P2,
  → plausible HLA binding.

- **Low / very negative score**
  → unfavorable amino acid,
  → peptide probably non-binding.

Each score must be **directly traceable**
to a matrix value.

---

### Edge cases to handle

- Peptide of length < 2
- Amino acid not present in the matrix
- Missing or corrupted PSSM file

In these cases:
- return a penalty or a documented neutral score,
- never crash the pipeline.

---

### Common pitfalls (to be absolutely avoided)

- ❌ Using the wrong indexing (off-by-one)
- ❌ Applying the matrix to the wrong position
- ❌ "Optimizing" the matrix
- ❌ Ignoring rare amino acids

---

### Limitations to state explicitly in the report

- The PSSM reflects experimental averages.
- It does not model 3D structure.
- Interactions with other positions are ignored.

---

### What this module does NOT claim to do

- It does not predict a real quantitative affinity.
- It does not replace NetMHCpan.
- It is not sufficient on its own to rank peptides.

It provides a **strong and interpretable signal**,
to be combined with the other HLA modules.

---

### Critical question you must be able to answer

> *"Why is this amino acid at position 2
> favored or penalized for this HLA allele,
> and how does the PSSM matrix encode this preference?"*

If you can clearly answer this question,
then you have understood the essence of peptide-HLA binding.
