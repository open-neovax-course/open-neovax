## Module C3 — Secondary anchors (additional positional contributions)

This module complements modules C1 and C2
by modeling the **secondary contributions** to peptide-HLA binding.

These are no longer "absolute" anchors,
but **positions that modulate the stability**
of the peptide-HLA complex.

---

### Biological context (intuition without required background)

Peptide-HLA binding is not binary.
Even if the main anchors (P2 and P9) are compatible:

- some peptides bind better than others,
- certain additional positions strengthen or weaken binding.

These positions:
- interact with the surface of the HLA groove,
- contribute to overall stability,
- are often called **secondary anchors**.

👉 They do not decide on their own,
but they **refine** binding.

---

### Biological hypothesis (explicitly stated)

Analyses of HLA ligands show that:
- certain positions other than P2 and P9
  exhibit compositional biases,
- these biases reflect favorable or unfavorable interactions
  with the HLA.

This module therefore implements the following hypothesis:

> *"The amino acids present at certain secondary positions
> contribute positively or negatively
> to the stability of the peptide-HLA complex."*

---

### Specific role of module C3

Unlike modules C1 and C2:
- C3 **does not decide** whether a peptide can bind,
- it **modulates** the quality of binding.

It allows:
- differentiating peptides with good main anchors,
- making the HLA score more gradual and realistic.

---

### Expected work (detailed logic)

#### 1. Select secondary positions

- Choose a subset of positions (typical examples: 1, 3, 7).
- The chosen positions must be:
  - explicitly listed,
  - briefly justified.

⚠️ You are not required to use all positions.
Fewer well-explained positions are better
than many without justification.

---

#### 2. Read the corresponding scores from the PSSM

- For each chosen position:
  - extract the corresponding amino acid,
  - read the score from the PSSM matrix.

You may:
- use the same matrix as C1/C2,
- or a simplified version if provided.

---

#### 3. Combine the scores

Accepted approaches:
- simple sum,
- average,
- weighted sum (if justified).

The goal is to produce:
- a single score,
- reflecting the overall effect of secondary anchors.

⚠️ The score must remain **subordinate**
to the main anchors.

---

### Inputs used

- `candidate.peptide_mut`

This module must **not** use:
- the WT peptide,
- the mutation position,
- information from other modules.

---

### Expected output (interpretation)

- **Slightly positive score**
  → favorable secondary positions,
  → complex stabilization.

- **Slightly negative score**
  → unfavorable secondary positions,
  → reduced stability.

This score must be:
- of smaller magnitude than C1/C2,
- interpretable position by position.

---

### Edge cases to handle

- Peptide too short for certain positions
- Amino acids absent from the matrix
- Chosen positions out of range

In these cases:
- ignore the affected position or apply a documented penalty,
- never crash the pipeline.

---

### Common pitfalls (to be absolutely avoided)

- ❌ Giving more weight to C3 than to C1/C2
- ❌ Using all positions without justification
- ❌ Duplicating the work of the main anchors
- ❌ Producing a score that is too dominant

---

### Limitations to state explicitly in the report

- The contributions of secondary positions are weak.
- Cooperative effects are not modeled.
- The PSSM is an approximation.

---

### What this module does NOT claim to do

- It does not replace the main anchors.
- It does not predict actual affinity.
- It does not model the complete structure of the complex.

It adds an **interpretable refinement**
to the overall HLA score.

---

### Critical question you must be able to answer

> *"Why were these secondary positions chosen,
> and how does their contribution improve
> the interpretation of the HLA score
> without overshadowing the role of the main anchors?"*

If you can clearly answer this question,
then your module is scientifically coherent.
