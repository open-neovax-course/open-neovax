# DEPARTMENT B — CELLULAR PROCESSING & PRESENTATION

This department focuses on **what happens BEFORE binding to HLA**.

Even a peptide that is perfect physicochemically (Department A)
and perfectly compatible with HLA (Department C)
may **never be presented** if it does not correctly follow
the intracellular pathway leading to MHC class I.

Department B therefore models the **constraints of the biological journey**
that a peptide must traverse to reach the HLA.

---

## General scientific objective

When a protein is produced in a cell, it does not automatically
become a peptide presented on the MHC.

It must follow a **precise pathway**:

1. Degradation of the protein by the **proteasome**
2. Generation of a peptide of compatible length
3. Transport of the peptide into the endoplasmic reticulum (ER) via **TAP**
4. Possible length adjustment by **ERAP**
5. Loading onto MHC-I

At each of these steps, **an implicit selection** occurs:
some peptides pass through, others do not.

---

## Key intuition (no biology background required)

You can think of this process as a **chain of successive filters**:

- the proteasome does not cut randomly,
- TAP does not transport all peptides with the same efficiency,
- ERAP does not process all N-termini in the same way.

👉 A peptide must be **compatible with all of these steps**
to be effectively presented.

Department B does not seek to simulate everything,
but to approximate these filters through **explicit heuristics**.

---

## What this department models (and what it does not)

### What Department B models
- **general preferences** observed experimentally,
- **robust trends** (e.g., certain C-termini are favored),
- qualitative constraints.

### What it does NOT model
- actual kinetics,
- 3D structure,
- detailed molecular interactions,
- fine cellular variations.

👉 The scores produced are **proxies**, not exact predictions.

---

## Overall role of Department B in Open-NeoVax

Department B plays the role of **biological feasibility filter**:

- Department A: *"Is this peptide physicochemically reasonable?"*
- **Department B: *"Can this peptide reasonably be generated and presented?"***
- Department C: *"Is this peptide compatible with the HLA?"*
- Department D: *"Is this peptide sufficiently non-self?"*

A peptide can:
- be excellent in A and C,
- but be penalized in B → low chance of being presented in vivo.

---

## Organization of Department B

The modules of this department each correspond
to a **distinct step of the intracellular pathway**.

Each group models **one link in the chain**.

### Overview of the groups

| Module | Biological step modeled | Question asked |
|------|----------------------------|---------------|
| B1 | Proteasome (C-terminal) | Can the peptide be generated as is? |
| B2 | TAP | Can the peptide be transported to the ER? |
| B3 | ERAP | Is the N-terminal compatible with trimming? |
| B4 | Sanity checks | Is the peptide biologically possible? |

---

## Scientific intuition common to Department B modules

All modules in this department are based on the same principle:

> *A presented peptide is the result of a compromise between several steps,
> each with its own preferences and biases.*

No module should:
- decide alone that a peptide is "good" or "bad",
- produce a harsh binary filter without justification.

Each module must provide:
- a **partial signal**,
- interpretable,
- combinable with others.

---

## Important message for students

You are not "simulating a cell".

You are:
- formalizing a **qualitative biological intuition**,
- transforming it into a **simple quantitative rule**,
- explicitly acknowledging its limitations.

This is exactly what many real pipelines do
when they lack complete models.

---

## Transition to the modules

The following sections detail each module of Department B,
explaining:
- the actual biological role of the step,
- the intuition used,
- the chosen proxy,
- and what is reasonable (or not) to expect from it.

*(Each module will be presented with the same level of detail as for Department A.)*
