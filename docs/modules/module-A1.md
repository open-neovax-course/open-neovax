## Module A1 — Hydrophobicity (Kyte–Doolittle)

This module is often the **first real contact** with a biochemical concept
in the project. It is deliberately simple, but **conceptually fundamental**.

---

### Context and intuition (no bio background required)

Peptides, like all proteins, exist in an **aqueous environment**.
Each amino acid has a stronger or weaker affinity for water:

- some prefer water → **hydrophilic**,
- others prefer to avoid water → **hydrophobic**.

A peptide is therefore characterized by an **overall hydrophobicity**, which influences:
- its solubility,
- its stability,
- its tendency to aggregate,
- its ability to be processed by the cell.

👉 **Neither hydrophobic extremes nor hydrophilic extremes are desirable.**

---

### Biological hypothesis (explicitly stated)

A peptide:
- **too hydrophobic** tends to aggregate or stick to membranes,
- **too hydrophilic** may be unstable or poorly compatible with the hydrophobic pockets of the MHC.

Experimentally observed HLA-I peptides generally exhibit
**intermediate hydrophobicity**, corresponding to a trade-off.

This module therefore implements the following hypothesis:

> *"Peptides whose overall hydrophobicity is intermediate are favored,
> while extreme values are penalized."*

---

### Tool used: the Kyte–Doolittle scale

The **Kyte–Doolittle** scale assigns to each amino acid
a numerical score representing its hydrophobic or hydrophilic character.

Examples (for reference):
- Isoleucine (I): highly hydrophobic
- Valine (V): hydrophobic
- Alanine (A): moderately hydrophobic
- Aspartate (D): hydrophilic
- Lysine (K): highly hydrophilic

This table is provided or can be coded as a Python dictionary.

---

### Expected work (detailed logic)

1. **Convert the sequence to numerical values**
  - for each amino acid in the mutant peptide,
  - retrieve the corresponding Kyte–Doolittle score.

2. **Calculate a global score**
  - average of the scores (recommended),
  - or sum normalized by the peptide length.

3. **Define an "acceptable" zone**

There is no universal "standard zone" of hydrophobicity in the state of the art.
Any zone defined in this module is a heuristic choice,
which must be biologically consistent, explainable, and acknowledged as such.
  - range of values considered optimal,
  - biologically justified (intermediate values).

4. **Penalize extremes**
  - score too high → penalty,
  - score too low → penalty,
  - intermediate score → score close to 0 (or slightly positive).

---

### Inputs used

- `candidate.peptide_mut`

> ⚠️ This module **must not** use the WT peptide.
> It focuses solely on the physical properties of the presented peptide.

---

### Expected output (interpretation)

- **Score close to 0**
 → reasonable hydrophobicity, physicochemically plausible peptide.

- **Negative score**
 → hydrophobicity too extreme (too hydrophobic or too hydrophilic).

The sign and magnitude of the score must be **interpretable** and explained in the report.

---

### Edge cases to handle explicitly

- Non-standard amino acids (`X`, `*`, etc.)
- Empty peptide
- Length outside the 8–11 aa range

In these cases:
- either return an explicit penalty,
- or a documented neutral score,
- but **never raise an exception**
