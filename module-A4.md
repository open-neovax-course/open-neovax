## Module A4 — Delta WT vs MUT (mutation impact)

This module is **conceptually central** in any neoantigen project.
It answers a simple but fundamental question:

> *"Does the mutation actually change something important in the peptide?"*

Without a significant change, there is generally **no neoepitope**.

---

### Context and intuition (no immunology background required)

A neoepitope is, by definition, **a mutant peptide**.
But not all mutations are equal:

- some mutations strongly alter the properties of the peptide,
- others are **nearly invisible** at the physicochemical level.

Examples:
- Leucine → Isoleucine
 → conservative mutation, little actual change
- Glycine → Tryptophan
 → non-conservative mutation, major change

👉 An interesting mutation is often one that **disrupts** the initial state.

---

### Biological hypothesis (explicitly stated)

A mutation that only slightly modifies the properties of the peptide:
- is often poorly distinguished from the WT peptide,
- is less likely to be recognized as non-self.

Conversely, a **non-conservative** mutation:
- alters the structure, charge, or steric bulk,
- increases the probability of immunological visibility.

This module therefore implements the following hypothesis:

> *"Mutations inducing a significant physicochemical change
> between the WT peptide and the mutant peptide are favored."*

---

### Expected work (detailed logic)

#### 1. Compare `peptide_wt` and `peptide_mut`

- Both peptides have the **same length**.
- The mutation must be located within the analyzed window.
- The comparison is performed **position by position**.

---

#### 2. Calculate a WT ↔ MUT distance

Simple accepted approaches:

##### a) Weighted Hamming distance
- 0 if AAs are identical
- 1 if AAs are different
- optionally weighted by position

##### b) Simplified substitution score (BLOSUM-like)
- conservative substitutions → low penalty
- radical substitutions → high penalty

You can:
- use a simplified matrix provided,
- or define amino acid classes (hydrophobic, charged, aromatic).

---

#### 3. Favor non-conservative mutations

- A conservative mutation should produce a low or neutral score.
- A non-conservative mutation should produce a higher score (or less penalized).

The score should reflect:
- the magnitude of the change,
- not simply the number of mutations.

---

### Inputs used

- `candidate.peptide_wt`
- `candidate.peptide_mut`
- `candidate.mut_pos_1based`

This module is the **first** to explicitly compare WT and MUT.

---

### Expected output (interpretation)

- **Score close to 0 or slightly negative**
 → conservative mutation, low impact.

- **More negative or higher score (depending on convention)**
 → non-conservative mutation, high impact.

The sign and scale must be clearly explained in the report.

---

### Edge cases to handle

- WT and MUT identical
- Mutation outside the window
- Different lengths
- Invalid amino acids

In these cases:
- return an explicit penalty,
- or a documented neutral score,
- but never crash the pipeline.

---

### Common pitfalls (to absolutely avoid)

- ❌ Only counting the number of mutations
- ❌ Ignoring the nature of the amino acids
- ❌ Over-interpreting a minor mutation
- ❌ Using a full BLOSUM matrix without justification

---

### Limitations to state explicitly in the report

- Actual immunological visibility also depends on the TCR.
- Some conservative mutations can be immunogenic.
- The score is a **proxy**, not a prediction.

---

### What this module does NOT claim to do

- It does not predict TCR recognition.
- It does not predict HLA binding.
- It is not sufficient on its own to define a neoepitope.

It provides a **signal of departure from self**,
intended to be combined with the HLA and safety modules.

---

### Critical question you must be able to answer

> *"How is the mutation modeled by your score
> sufficiently different from the WT to justify immunological interest?"*

If you can clearly answer this question,
then your module correctly captures the spirit of the problem.
