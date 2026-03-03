# DEPARTMENT C — HLA (CORE OF THE PROJECT)

This department constitutes the **scientific core** of Open-NeoVax.

It answers the central question of any neoantigen project:

> *"Can this peptide plausibly bind
> to the patient's HLA and be presented on the cell surface?"*

Without peptide-HLA compatibility,
there is **neither presentation** nor recognition by T lymphocytes.

---

## General scientific objective

Department C aims to **approximate peptide-HLA compatibility**
using **explicit and interpretable models**,
based on position-specific preference matrices.

Unlike modern deep learning approaches:
- the rules are visible,
- each position explicitly contributes to the score,
- decisions are traceable.

The goal is not to outperform NetMHCpan,
but to understand **why** a peptide is favored or penalized.

---

## Biological intuition (no immunology background required)

HLA class I molecules:
- have a **binding groove**,
- impose strict constraints on the peptide,
- primarily recognize certain key positions called **anchors**.

For a typical HLA-I peptide (9-mer):
- position 2 (P2),
- and the C-terminal position (P9),

are often **critical** for binding.

👉 If these positions are not compatible,
the peptide **does not bind**, even if it is perfect elsewhere.

---

## Why explicit matrices?

Historically, before deep learning,
peptide-HLA binding prediction relied on:

- **PSSMs** (Position-Specific Scoring Matrices),
- built from known ligands,
- that indicate, for each position,
 which amino acids are favored or disfavored.

These matrices:
- are simple,
- interpretable,
- pedagogically ideal.

They allow one to say:
> *"This peptide is penalized because its amino acid at position 2
> is unfavorable for this HLA allele."*

---

## Overall role of Department C in Open-NeoVax

In the pipeline logic:

- Department A: general properties of the peptide
- Department B: feasibility of generation and transport
- **Department C: structural compatibility with HLA**
- Department D: safety and similarity to self

Department C is:
- **the dominant signal** in the final score,
- the one that most strongly discriminates between peptides.

Without a reasonable HLA score,
the other modules have little relevance.

---

## General organization of Department C

The modules of Department C decompose peptide-HLA binding
into **independent positional contributions**.

Each group models a part of the HLA groove.

### Conceptual overview

| Module | Aspect modeled | Question asked |
|------|------------------|----------------|
| C1 | P2 anchor | Is the residue at position 2 compatible? |
| C2 | P9 anchor (C-term) | Is the C-terminal compatible? |
| C3 | Secondary anchors | Are the secondary positions favorable? |
| C4 | Delta WT vs MUT (binding) | Does the mutation alter HLA affinity? |
| (Bonus) | Multi-allele | Is the peptide compatible with multiple HLAs? |

---

## Scientific intuition common to Department C modules

All modules are based on the same fundamental principle:

> *"Peptide-HLA binding is the sum of local contributions,
> dominated by a few key positions."*

Each module:
- produces a partial score,
- interpretable position by position,
- combinable with others.

No module:
- "decides" alone,
- should be opaque,
- should mimic a neural network.

---

## Key assumptions (and their limitations)

Assumptions made:
- HLA preferences can be approximated by matrices,
- positions are partially independent,
- known ligands reflect biological preferences.

Known limitations:
- 3D structure is not modeled,
- interactions between positions are ignored,
- actual affinity depends on cellular context.

These limitations must be **known and acknowledged** by students.

---

## Key message for students

You are coding:
- **the explanatory core** of the pipeline,
- the part closest to actual historical methods,
- a simple but conceptually powerful model.

If you understand why a peptide
is favored or penalized by **each position**,
then you have understood the essence
of peptide-HLA prediction.

---

## Transition to the modules

The following sections detail each module of Department C,
with for each:
- the biological role of the position,
- how to read the matrix,
- the scoring logic,
- and pitfalls to avoid.

*(Each module is detailed with the same level of rigor
as those of Departments A and B.)*
