# DEPARTMENT A — PHYSICOCHEMICAL PROPERTIES OF THE PEPTIDE

This department constitutes **the physicochemical foundation** of the Open-NeoVax project.
It focuses solely on the **peptide itself**, independently of:
- the HLA allele,
- the immune system,
- the cellular context.

The objective is to reason like a biochemist:
before even asking *"which HLA could this peptide bind to"*, one must ask
**whether the peptide has plausible physicochemical properties to exist, be stable, and be handled by the cell.**

---

## General scientific objective

Peptides presented by MHC class I are **not arbitrary**.
They obey a set of simple physical constraints:

- they must be **sufficiently stable** in solution,
- they must not **aggregate** spontaneously,
- they must not be **extreme** (too hydrophobic, too charged, too repetitive),
- they must have a **composition compatible** with an aqueous and proteinaceous environment.

This department models these constraints through **simple proxies**, deliberately unsophisticated but biologically defensible.

---

## Central question addressed by this department

> **"Is this peptide physicochemically reasonable as an HLA ligand,
> even before considering HLA presentation or TCR recognition?"**

In other words:
- if this peptide were generated in a cell,
- would it have properties compatible with:
 - a transient existence in solution,
 - handling by cellular proteins,
 - subsequent presentation?

---

## Why this department matters (key intuition)

A peptide can:
- be **perfect for HLA** on paper,
- but be **biologically absurd** because of its physical properties.

Examples:
- an extremely hydrophobic peptide can **aggregate** and never reach the MHC,
- an extremely charged peptide can be **unstable**,
- a highly repetitive peptide can be **biologically uninformative**,
- a mutation that changes almost nothing physically is often **immunologically invisible**.

👉 Department A therefore acts as a **basic plausibility filter**.

---

## What you need to remember (no biology background required)

Even without a biology background, you can reason with these simple ideas:

- Proteins and peptides exist in **water**.
- Extremes are rarely good:
 - too hydrophobic → aggregation,
 - too hydrophilic → instability,
 - too charged → non-specific interactions.
- An "interesting" mutation must **change something** measurable.

These principles are sufficient to understand and implement all modules in this department.

---

## General logic of Department A modules

All modules in Department A follow the same philosophy:

1. **Extract a simple property** from the mutant peptide
  (hydrophobicity, charge, diversity, WT/MUT distance, etc.).

2. **Compare this property to a reasonable range**
  (intermediate values preferred, extremes penalized).

3. **Return a continuous and interpretable score**, not a binary decision.

---

## What Department A modules do NOT do

- They do **not** predict HLA binding.
- They do **not** model the TCR.
- They make **no final decision**.

They provide **weak but informative signals**,
which will be combined with those from other departments.

---

## Overall vision (keep in mind)

You can think of Department A as:

> **"The physicochemical sanity check of a peptide."**

If a peptide fails here:
- it can sometimes be rescued by a very strong HLA score,
- but it starts with a clear biological handicap.

---

## Transition to the modules

The modules in Department A will now explore different facets
of this physicochemical plausibility:

- overall hydrophobicity,
- charge and solubility,
- sequence complexity,
- actual impact of the mutation compared to the wild-type peptide.

Each module corresponds to **a distinct biological intuition**,
which you will need to understand, implement, and critically evaluate.

*(The scientific and technical details of each module are presented in the following sections.)*
