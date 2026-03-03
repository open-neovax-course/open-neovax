# DEPARTMENT D — SAFETY & SELF-SIMILARITY

This department constitutes the **safeguard** of the Open-NeoVax project.

It answers a question that the other departments do not ask:

> *"Is this peptide sufficiently different from self
> to be a safe and potentially immunogenic vaccine candidate?"*

Without this verification, the pipeline could recommend
peptides that are **dangerous** (auto-reactivity)
or **useless** (indistinguishable from self).

---

## General scientific objective

Departments A, B, and C evaluate the physicochemical properties,
presentation feasibility, and HLA compatibility of a peptide.

But none of them checks whether the mutant peptide
**resembles the repertoire of normal human peptides**.

Yet, a peptide that:
- is **identical** to a self peptide -> risk of autoimmunity,
- is **very similar** to self -> low immunogenicity,
- **does not actually incorporate the mutation** -> is not a neoepitope.

Department D models these verifications
through **explicit comparisons** and **consistency checks**.

---

## Key intuition (no immunology background required)

The immune system is **educated** to tolerate normal peptides from the body.

An effective vaccine must target something that the immune system
perceives as **foreign** (non-self).

If the mutant peptide is:
- identical to a normal peptide -> the immune system ignores it
  (or worse, there is a risk of autoimmune reaction),
- very different from self -> it has a greater chance of being recognized as new.

Department D therefore verifies that the vaccine candidate is:
1. **different from self** (safety),
2. **truly mutant** (consistency).

---

## What this department models (and what it does not)

### What Department D models
- similarity to self (exact and approximate),
- consistency of the mutation within the peptide window,
- elementary safety signals.

### What it does NOT model
- actual immunological tolerance,
- complex mechanisms of thymic selection,
- clinical effects of an autoimmune reaction,
- actual immunogenicity (which also depends on the TCR).

The scores produced are **proxies for safety and relevance**,
not clinical predictions.

---

## Overall role of Department D in Open-NeoVax

In the pipeline logic:

- Department A: *"Is this peptide physicochemically reasonable?"*
- Department B: *"Can this peptide be generated and presented?"*
- Department C: *"Is this peptide compatible with HLA?"*
- **Department D: *"Is this peptide sufficiently non-self and consistent?"***

A peptide can:
- be excellent in A, B, and C,
- but be disqualified in D -> too similar to self or inconsistent mutation.

Department D acts as a **final safety filter**.

---

## Organization of Department D

The modules of this department each correspond
to a **distinct verification**.

### Overview of the modules

| Module | Verification | Question asked |
|--------|-------------|---------------|
| D1 | Exact similarity to self | Is the peptide identical to a known human peptide? |
| D2 | Approximate similarity (optional) | Is the peptide too close to a human peptide? |
| D3 | Mutation within the window | Is the mutation actually present in the peptide? |

---

## Scientific intuition common to Department D modules

All modules are based on the same principle:

> *A good vaccine candidate must be sufficiently non-self
> to be recognized by the immune system,
> and its mutation must be actually included in the analyzed peptide.*

Each module:
- produces a safety or consistency signal,
- directly interpretable,
- combinable with others.

---

## Data used by Department D

Unlike the other departments,
Department D uses **external reference data**:

- **Corpus of human peptides** (`human_peptides_small.txt`):
  a list of peptides derived from the normal human proteome,
  provided in the `data/` folder.
- **Both peptides** (WT and MUT) of the candidate.

These data must be loaded locally and robustly.

---

## Important message for students

You are not "predicting immune tolerance".

You are:
- formalizing a **simple safety verification**,
- transforming it into an **explicit quantitative rule**,
- explicitly acknowledging its limitations.

This is exactly what real pipelines do
for neoantigen vaccine design.

---

## Transition to the modules

The following sections detail each module of Department D,
explaining:
- the biological role of the verification,
- the comparison logic,
- the chosen proxy,
- and what is reasonable (or not) to expect from it.

*(Each module is presented with the same level of detail
as those of Departments A, B, and C.)*
