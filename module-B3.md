## Module B3 — ERAP trimming (N-terminal proxy)

This module models a **frequently overlooked but important** step
in MHC-I processing: **N-terminal trimming** by the ERAP enzymes
(Endoplasmic Reticulum Aminopeptidases).

Even after TAP transport, a peptide can be **shortened or destroyed**
before being loaded onto the HLA.

---

### Biological context (intuition without required background)

Peptides transported by TAP are not always
of the ideal length (often 9 aa for HLA-I).

In the endoplasmic reticulum:
- enzymes called **ERAP1 / ERAP2**
- progressively cleave the **N-terminal** end of the peptide
- until a length compatible with the MHC is reached.

This process:
- strongly depends on the **first amino acid**,
- can be efficient, slow, or blocked depending on the N-terminal.

👉 ERAP acts as an **adjuster**, but also as a **filter**.

---

### Biological hypothesis (explicitly stated)

Experimental observations show that:
- certain N-terminal residues are **less efficiently trimmed**,
- which can prevent the peptide from reaching the correct length
  or lead to its degradation.

This module therefore implements the following hypothesis:

> *"Peptides whose N-terminal is unfavorable for ERAP trimming
> have a reduced probability of being presented by MHC-I."*

---

### What this module models (and what it does not)

This module does not model actual ERAP trimming.
It penalizes certain N-terminal residues that are rarely observed
among HLA-I ligands, assuming that these biases
partly reflect ERAP preferences.

#### Modeled (proxy)
- General ERAP preferences regarding the **first amino acid**.

#### Not modeled
- The actual initial length of the precursor,
- The successive trimming steps,
- The differences between ERAP1 vs ERAP2,
- Enzymatic kinetics.

👉 Once again: **simple proxy**, acknowledged as such.

---

### Expected work (detailed logic)

#### 1. Examine the first amino acid

- Extract the first character of `candidate.peptide_mut`.
- Verify that it is a valid amino acid.

---

#### 2. Penalize certain residues if justified

Possible approaches:
- define a list of disfavored N-terminal residues,
- use a simple table (favored / neutral / penalized),
- penalize bulky or charged residues if you justify it.

⚠️ The choice of penalized residues must be:
- consistent,
- explicitly justified,
- acknowledged as heuristic.

---

#### 3. Implement a clear and interpretable score

The score must:
- depend **solely** on the N-terminal,
- be continuous or discrete but documented,
- be easily interpretable biologically.

Examples:
- fixed penalty if AA ∈ disfavored list,
- negative score proportional to an unfavorable category.

---

### Inputs used

- `candidate.peptide_mut`

This module must **not** use:
- the WT peptide,
- the C-terminal,
- the HLA,
- other modules.

---

### Expected output (interpretation)

- **Score close to 0**
  → N-terminal compatible with plausible trimming.

- **Negative score**
  → Unfavorable N-terminal, potentially inefficient trimming.

---

### Edge cases to handle

- Empty peptide
- Length < 1
- Non-standard amino acids

In these cases:
- return an explicit penalty or a documented neutral score,
- never crash the pipeline.

---

### Common pitfalls (to be absolutely avoided)

- ❌ Trying to model the entire trimming process
- ❌ Using the entire sequence
- ❌ Implementing overly complex rules
- ❌ Confusing ERAP with the proteasome

---

### Limitations to state explicitly in the report

- ERAP trimming depends on the actual precursor.
- The N-terminal alone is partial information.
- This score is a **very coarse proxy**.

---

### What this module does NOT claim to do

- It does not simulate ERAP.
- It does not predict the actual final length.
- It does not guarantee presentation.

It provides a **complementary signal**
to the proteasome and TAP.

---

### Critical question you must be able to answer

> *"Why is the N-terminal of this peptide
> considered favorable or unfavorable
> in your model,
> and what are the limitations of this reasoning?"*

If you can clearly answer this question,
then your module correctly captures the expected biological intuition.
