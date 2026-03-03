## Module A3 — Sequence complexity / entropy

This module introduces a key concept often overlooked in naive bioinformatics:
**sequence complexity**.

It is not about chemistry, but about **informational content**.

---

### Context and intuition (accessible without bio background)

A peptide is a **chain of symbols** (amino acids).
Some chains are rich in information, others very poor.

Examples:
- `LLFGYPRLV` → varied sequence, informative
- `LLLLLLLLL` → repetitive sequence, poorly informative
- `AAAAAAAAV` → minimal variation

Even without biology, the intuition is immediate:
- a highly repetitive sequence **carries little information**,
- it is often the result of artificial constraints,
- it is rarely enriched among biologically selected peptides.

---

### Biological hypothesis (explicitly stated)

Peptides presented by MHC-I that are observed experimentally show
a **minimal diversity** of residues.

Extremely repetitive peptides:
- are poorly specific,
- are often biologically disfavored,
- have limited immunological potential.

This module therefore implements the following hypothesis:

> *"Peptides exhibiting low sequence complexity
> (high repetition of amino acids)
> are penalized compared to more diverse peptides."*

---

### Concept of complexity / entropy (simplified version)

The **complexity of a sequence** can be seen as:
- the number of distinct symbols,
- their distribution,
- the unpredictability of the sequence.

In this project, the goal is **not** to perform advanced information theory,
but to implement a **simple and robust proxy**.

---

### Expected work (detailed logic)

#### 1. Measure amino acid diversity

Several simple approaches are accepted:

- number of distinct amino acids,
- maximum frequency of a single amino acid,
- proportion of the dominant residue.

Examples:
- `LLLLLLLLL` → 1 distinct AA → very low complexity
- `LLFGYPRLV` → 7–8 distinct AAs → good complexity

---

#### 2. Implement a simple complexity score

Accepted approaches:
- ratio `nb_distinct_AAs / length`,
- simplified Shannon entropy,
- penalty based on the frequency of the dominant residue.

The important thing is that:
- the score increases with diversity,
- the score decreases with repetition.

---

#### 3. Penalize peptides that are too repetitive

- define a threshold or a penalty function,
- progressively penalize sequences dominated by a single AA,
- avoid a harsh binary decision if possible.

---

### Inputs used

- `candidate.peptide_mut`

This module must not use:
- the WT peptide,
- the HLA,
- the cellular context.

---

### Expected output (interpretation)

- **Score close to 0**
 → sufficiently complex sequence, no penalty.

- **Negative score**
 → overly repetitive sequence, low informational content.

The link between score and complexity must be clear and explainable.

---

### Edge cases to handle

- Empty peptide
- Length outside the 8–11 aa range
- Invalid amino acids

In these cases:
- return an explicit penalty or a documented neutral score,
- never raise an uncontrolled exception.

---

### Common pitfalls (to avoid)

- ❌ Artificially favoring highly varied sequences
- ❌ Implementing needlessly complex entropy
- ❌ Producing a score that is difficult to interpret
- ❌ Confusing complexity with hydrophobicity

---

### Limitations to state explicitly in the report

- Complexity is not a direct measure of immunogenicity.
- Biologically valid peptides can have low complexity.
- The score is an **informational proxy**, not a biological truth.

---

### What this module does NOT claim to do

- It does not predict HLA binding.
- It does not model TCR recognition.
- It does not replace other physicochemical criteria.

It provides a **complementary signal**, useful for detecting
aberrant or artificially extreme peptides.

---

### Critical question you must be able to answer

> *"Why is this peptide penalized by your complexity score,
> and how is this penalty biologically and informationally reasonable?"*

If you can clearly answer this question,
then your module fulfills its role.
