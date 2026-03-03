## Module C4 — Delta binding WT vs MUT

This module is the **last pillar** of department C.
It answers a direct and highly informative question:

> *"Does the mutation improve or degrade
> the peptide's compatibility with the HLA?"*

This signal is one of the most discriminating
for identifying a relevant neo-epitope.

---

### Biological context (intuition with no background required)

Modules C1, C2, and C3 evaluate the **mutant peptide alone**.

But an interesting neo-epitope is not just a peptide
that binds well to the HLA:
it is a peptide that binds **better** (or at least as well)
than the wild-type (WT) peptide.

Why?
- If the mutation **degrades** HLA binding,
  the mutant peptide will be less presented than the WT.
- If the mutation **improves** binding,
  the mutant peptide will be preferentially presented.

This module therefore directly compares
the HLA score of the mutant to the HLA score of the wild-type.

---

### Biological hypothesis (explicitly stated)

Analyses show that the most immunogenic neo-epitopes
are often those whose mutation **improves** or **maintains**
HLA binding relative to the WT peptide.

This module therefore implements the following hypothesis:

> *"A mutant peptide whose HLA score is higher
> than that of the WT peptide is a better neo-epitope candidate
> than a peptide whose mutation degrades binding."*

---

### Specific role of module C4 in department C

Unlike modules C1–C3:
- C4 does **not** focus on a single position,
- it compares **two peptides** (WT and MUT)
  using the same scoring logic,
- it produces a **difference** signal (delta).

This delta is highly informative:
- it allows distinguishing neutral mutations
  from mutations that actually change presentation.

---

### Expected work (detailed logic)

#### 1. Compute an HLA score for the MUT peptide

- Use the provided PSSM.
- Sum the scores across all positions (or key positions)
  for `candidate.peptide_mut`.

#### 2. Compute the same HLA score for the WT peptide

- Apply exactly the same logic to `candidate.peptide_wt`.

#### 3. Return the delta

- `delta = score_mut - score_wt`
- A positive delta means the mutation **improves** binding.
- A negative delta means the mutation **degrades** binding.
- A delta close to 0 means the mutation is neutral for the HLA.

The returned score is directly this delta.

---

### Inputs used

- `candidate.peptide_mut`
- `candidate.peptide_wt`

This module is the **only module in department C**
that explicitly uses the WT peptide.

---

### Expected output (interpretation)

- **Positive score**
  -> the mutation improves HLA binding,
  -> potentially interesting candidate.

- **Score close to 0**
  -> neutral mutation from the HLA perspective.

- **Negative score**
  -> the mutation degrades binding,
  -> less interesting candidate.

The score must be directly interpretable
as an estimated affinity difference.

---

### Edge cases to handle

- WT and MUT identical (delta = 0)
- Different lengths between WT and MUT
- Amino acids absent from the PSSM
- Missing or corrupted PSSM file

In these cases:
- return a documented neutral score (e.g., 0.0),
- never crash the pipeline.

---

### Common pitfalls (to absolutely avoid)

- Not using the same scoring method for WT and MUT
- Reversing the delta direction (score_wt - score_mut instead of score_mut - score_wt)
- Using only certain positions without documenting it
- Duplicating the work of C1/C2/C3 without providing the comparison

---

### Limitations to state explicitly in the report

- The HLA delta does not measure TCR immunogenicity.
- The PSSM is a statistical approximation.
- A good delta does not guarantee actual presentation.
- Interactions between positions are ignored.

---

### What this module does NOT claim to do

- It does not predict TCR recognition.
- It is not sufficient to identify a neo-epitope.
- It does not model the structural effects of the mutation.

It provides an **HLA affinity change signal**,
to be combined with anchors and the other departments.

---

### Critical question you must be able to answer

> *"How is the HLA score difference between the mutant peptide
> and the wild-type peptide informative
> for identifying a relevant neo-epitope?"*

If you can clearly answer this question,
then your module correctly captures
one of the most important signals in the pipeline.
