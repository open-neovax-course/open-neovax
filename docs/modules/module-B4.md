## Module B4 — Sanity checks (minimal biological validity)

This module is **intentionally simple**, but **fundamental**.
It does not seek to finely score a peptide,
but to eliminate (or heavily penalize) **biologically impossible** cases.

It is a **safeguard** module.

---

### Context and general intuition

In any real bioinformatics pipeline, a significant part of the work
involves handling **incorrect or inconsistent inputs**:

- parsing errors,
- incomplete data,
- poorly formatted data,
- experimental artifacts.

Without minimal checks:
- absurd peptides can be favored,
- scores become difficult to interpret,
- the pipeline loses all credibility.

👉 This module ensures that **only plausible peptides**
can be ranked highly.

---

### Biological hypothesis (explicitly stated)

A peptide that:
- does not have a length compatible with MHC-I,
- contains non-biological characters,
- violates elementary constraints,

**cannot** be a valid HLA-I ligand,
regardless of its quality on other criteria.

This module therefore implements the following hypothesis:

> *"Biologically impossible peptides must be
> heavily penalized, regardless of all other scores."*

---

### Specific role of module B4 in department B

Unlike modules B1–B3:
- B4 **does not model a specific biological step**,
- it acts as a **global consistency filter**.

It protects the pipeline against:
- invalid inputs,
- aberrant behaviors,
- silent bugs.

---

### Expected work (detailed logic)

#### 1. Verify peptide length

- The peptide must have a length between **8 and 11 amino acids**.
- Outside this range:
  - the peptide is considered invalid for this project.

Justification:
- MHC-I ligands are predominantly 8–11 aa,
- anything else is out of scope.

---

#### 2. Verify the amino acid alphabet

- The peptide must contain **only** the 20 standard AAs:
  `ACDEFGHIKLMNPQRSTVWY`
- Any other character (`X`, `*`, digits, etc.) is invalid.

Justification:
- these characters do not correspond to real peptides,
- their presence indicates a data error.

---

#### 3. Return a strong penalty if invalid

- If **at least one** condition fails:
  - return a strong and explicit penalty.
- The score must dominate the other modules
  (e.g., very negative value).

⚠️ Important:
- the module **must not raise an unhandled exception**,
- it must return a numerical score.

---

### Inputs used

- `candidate.peptide_mut`

This module must **not** use:
- the WT peptide,
- the mutation position,
- the HLA,
- scores from other modules.

---

### Expected output (interpretation)

- **Neutral score or close to 0**
  → peptide is formally valid.

- **Very negative score**
  → invalid peptide, must be rejected by the final ranking.

This score is **not finely biologically interpretable**:
it encodes a hard validity rule.

---

### Edge cases to handle explicitly

- Empty peptide
- `None` string
- Invalid characters
- Inconsistent length

All these cases must be handled **without crashing**.

---

### Common pitfalls (to be absolutely avoided)

- ❌ Raising an exception that breaks the entire pipeline
- ❌ Returning `NaN` or `None`
- ❌ Applying a weak penalty
- ❌ Redefining complex biological rules

---

### Limitations to state explicitly in the report

- This module does not distinguish "bad" from "good" peptides,
  only "possible" from "impossible".
- It does not replace any other scoring module.

---

### What this module does NOT claim to do

- It does not predict HLA presentation.
- It does not predict immunogenicity.
- It does not rank valid peptides against each other.

It serves solely to **avoid biological absurdities**.

---

### Critical question you must be able to answer

> *"Why is this peptide considered invalid
> in your module,
> and why is it reasonable to heavily penalize it
> without looking at the other scores?"*

If you can clearly answer this question,
then your module perfectly fulfills its role.
