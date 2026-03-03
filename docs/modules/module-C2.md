## Module C2 — C-terminal anchor (P9 / PΩ)

This module complements module C1 and models **the other major anchor**
of peptide-HLA class I binding:
the **C-terminal position** of the peptide (often denoted P9 or PΩ).

Together with position 2, the C-terminal constitutes
the **structural foundation** of HLA binding.

---

### Biological context (intuition without required background)

An HLA-I peptide (9-mer) is held in the HLA groove
by two main anchor points:

- the N-terminal end (via P2),
- the C-terminal end (P9 / PΩ).

The C-terminal:
- inserts deeply into a hydrophobic pocket of the HLA,
- is **extremely constrained**,
- tolerates few variations depending on the allele.

👉 If the C-terminal is incompatible,
the peptide **does not bind**, even if everything else is optimal.

---

### Biological hypothesis (explicitly stated)

Known HLA-I ligands show
very strong preferences for certain amino acids
at the C-terminal position,
specific to each HLA allele.

This module therefore implements the following hypothesis:

> *"A peptide whose C-terminal matches the preferences
> of the considered HLA allele
> has a much higher probability of binding to the HLA."*

This hypothesis is **one of the most robust**
in all of peptide immunology.

---

### Why this module is distinct from C1

Even though the logic is similar to C1:

- P2 and P9 correspond to **different pockets**,
- their amino acid preferences are **independent**,
- their impact on binding is **complementary**.

Separating the two modules allows:
- better interpretability,
- clear positional analysis,
- easy extension to other positions.

---

### Expected work (detailed logic)

#### 1. Load the same provided PSSM matrix

- Reuse the provided HLA PSSM matrix (e.g., `hla_pssm_A0201.csv`).
- The column corresponding to the C-terminal position
  (usually position 9) will be used.

⚠️ Important:
- you **should not** reload the matrix at each call
  if you can avoid it (load at the module level).

---

#### 2. Extract the C-terminal amino acid

- Extract the **last character** of `candidate.peptide_mut`.
- Verify:
  - that the length is sufficient,
  - that the amino acid is valid.

---

#### 3. Return the corresponding score

- Read from the PSSM the value associated with:
  - this amino acid,
  - column P9 (or equivalent).
- Return this score directly,
  or after a documented transformation.

The score must be:
- traceable,
- interpretable,
- directly linked to the matrix.

---

### Inputs used

- `candidate.peptide_mut`

This module must **not** use:
- the WT peptide,
- the mutation position,
- scores from other modules,
- multi-position logic.

---

### Expected output (interpretation)

- **High / slightly negative score**
  → favorable C-terminal,
  → plausible HLA anchoring.

- **Low / very negative score**
  → unfavorable C-terminal,
  → HLA binding unlikely.

The score must be explainable by:
> *"This amino acid is (dis)favored at the C-terminal
> for this HLA allele."*

---

### Edge cases to handle

- Peptide of length < 1
- Amino acid absent from the matrix
- Length ≠ 9 (if you choose to accept 8–11)

In these cases:
- return an explicit penalty or a documented neutral score,
- never crash the pipeline.

---

### Common pitfalls (to be absolutely avoided)

- ❌ Confusing P9 with P8 (indexing error)
- ❌ Using the wrong matrix column
- ❌ Combining multiple positions in this module
- ❌ Artificially smoothing the scores

---

### Limitations to state explicitly in the report

- C-terminal preferences strongly depend on the allele.
- The PSSM is a statistical approximation.
- Cooperative effects between positions are not modeled.

---

### What this module does NOT claim to do

- It does not predict an exact quantitative affinity.
- It does not model 3D structure.
- It does not replace modern ML models.

It provides a **major anchoring signal**,
to be combined with P2 and secondary anchors.

---

### Critical question you must be able to answer

> *"Why is the C-terminal of this peptide
> favorable or unfavorable for this HLA allele,
> and why is this position crucial
> for the stability of the peptide-HLA complex?"*

If you can clearly answer this question,
then you have mastered one of the central concepts
of HLA-I presentation.
