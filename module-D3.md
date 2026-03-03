## Module D3 — Mutation within the window

This module is a **fundamental consistency check**.
It verifies that the mutation is actually present
in the analyzed peptide.

Without this verification, the pipeline could score and recommend
a peptide **that is not actually a neo-epitope**.

---

### Context and intuition (no bio background required)

A neo-epitope is, by definition, a peptide **containing the mutation**.

But in the input data, it is possible that:
- the declared mutation position is **outside the window** of the peptide,
- the WT peptide and the MUT peptide are **identical**
  (data error or silent mutation),
- the declared position does not correspond to an actual difference
  between WT and MUT.

In all these cases, the peptide is **not a neo-epitope**
and should not be recommended as a vaccine candidate.

---

### Biological hypothesis (explicitly stated)

A peptide without an included mutation is not a neo-epitope.

This module therefore implements the following hypothesis:

> *"A candidate peptide whose mutation is not actually
> contained within the analyzed window
> must be strongly penalized,
> because it is indistinguishable from the wild-type peptide."*

---

### Specific role of module D3

Unlike modules D1 and D2:
- D3 does **not** compare to external self,
- it verifies the **internal consistency** of the candidate.

It acts as a **data quality filter**,
essential in any real bioinformatics pipeline.

---

### Expected work (detailed logic)

#### 1. Verify that WT and MUT are different

- Compare `candidate.peptide_wt` and `candidate.peptide_mut`.
- If they are identical, the candidate is invalid.

---

#### 2. Verify the declared mutation position

- Read `candidate.mut_pos_1based`.
- Verify that this position is:
  - a valid integer,
  - between 1 and the length of the peptide.

---

#### 3. Verify consistency between position and sequences

- At position `mut_pos_1based`:
  - the amino acid in `peptide_wt` and `peptide_mut`
    **must be different**.
- If the amino acid is identical at this position,
  the mutation declaration is inconsistent.

---

#### 4. Return a penalty if inconsistent

- If **all checks pass**:
  return a neutral score (e.g., 0.0).
- If **at least one check fails**:
  return a strong penalty.

The penalty must be strong enough
so that the peptide is ranked last.

---

### Inputs used

- `candidate.peptide_wt`
- `candidate.peptide_mut`
- `candidate.mut_pos_1based`

This module is the **only one** to use all three fields simultaneously.

---

### Expected output (interpretation)

- **Neutral score (0.0)**
  -> mutation present and consistent, valid candidate.

- **Very negative score**
  -> inconsistency detected, invalid candidate.

The score is essentially binary (valid / invalid),
expressed as a float for compatibility with the pipeline.

---

### Edge cases to handle

- Empty peptide
- Missing WT or MUT
- `mut_pos_1based` = 0 or negative
- `mut_pos_1based` > peptide length
- Different lengths between WT and MUT

In these cases:
- return an explicit penalty,
- never crash the pipeline.

---

### Common pitfalls (to absolutely avoid)

- Forgetting that `mut_pos_1based` is indexed starting from 1
  (not from 0 as in Python)
- Only checking one of the three conditions
- Returning a penalty that is too weak
- Not handling cases where lengths differ

---

### Limitations to state explicitly in the report

- This module does not verify whether the mutation is biologically real.
- Upstream data errors (parsing, peptide generation)
  are not corrected, only detected.
- A consistent mutation does not guarantee immunogenicity.

---

### What this module does NOT claim to do

- It does not predict immunogenicity.
- It does not correct data.
- It does not replace the similarity filters (D1, D2).

It only guarantees that **the candidate is a true neo-epitope**
from a data perspective.

---

### Critical question you must be able to answer

> *"Why can a peptide whose mutation is not included
> in the analyzed window
> not be considered a neo-epitope,
> and what checks does your module perform
> to detect this inconsistency?"*

If you can clearly answer this question,
then your module perfectly fulfills its role
as a consistency check.
