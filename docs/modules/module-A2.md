## Module A2 — Net charge / pI (proxy)

This module naturally complements module A1 (hydrophobicity) by addressing
another **fundamental physicochemical property** of peptides:
their **overall electric charge**.

The goal is not to perform detailed chemistry,
but to reason with simple, robust, and interpretable concepts.

---

### Context and intuition (no bio background required)

Peptides are chains of amino acids immersed in an aqueous medium.
Some amino acids carry an **electric charge**:

- **positively charged** (e.g., Lysine, Arginine),
- **negatively charged** (e.g., Aspartate, Glutamate),
- **neutral**.

The **overall charge** of a peptide strongly influences:
- its **solubility**,
- its **stability**,
- its nonspecific interactions with other proteins,
- its behavior in a dense cellular environment.

**Extremes are rarely favorable**.

---

### Biological hypothesis (explicitly stated)

A peptide:
- **highly positively charged** may interact nonspecifically
  with membranes or proteins,
- **highly negatively charged** may be unstable or disfavored
  during presentation.

Experimentally observed HLA-I peptides tend to have
a **moderate overall charge**, often close to neutrality.

This module therefore implements the following hypothesis:

> *"Peptides whose overall charge is moderate are favored,
> while extreme charges (positive or negative) are penalized."*

---

### Concept of pI (conceptual reminder)

The **isoelectric point (pI)** is the pH at which a molecule is electrically neutral.
Its exact calculation:
- depends on pH,
- involves several chemical constants,
- is **out of scope** for this project.

In Open-NeoVax, the real pI is **not** used,
but rather a **simple proxy** based on net charge.

---

### Expected work (detailed logic)

#### 1. Define a charge table per amino acid

You must define (or reuse) a simple table, for example:

##### Recommended charge table (standard)

Unless explicitly justified otherwise, the following charge table should be used:

- K (Lysine): +1
- R (Arginine): +1
- D (Aspartate): −1
- E (Glutamate): −1
- All other amino acids: 0

The case of histidine (H) can be treated as:
- either neutral (0),
- or weakly positive (+1),
provided the choice is documented.

This table is a **pedagogical standard** commonly used
to estimate the net charge of peptides.


This table is a **modeling choice** and must be documented.

---

#### 2. Calculate an overall charge

From the mutant peptide:
- sum the charges of each amino acid,
- optionally normalize by the peptide length.

The result is a **net charge proxy**, not an exact physicochemical value.

---

#### 3. Define a preferred neutral zone

- there is no universal threshold,
- the "ideal" zone corresponds to a moderate charge.

Examples of accepted strategies:
- charge close to 0 is favored,
- increasing penalty with |charge|,
- central interval defined a priori.

The important thing is that the choice is:
- consistent,
- explainable,
- acknowledged as heuristic.

---

#### 4. Penalize extremes

Very positive or very negative charges must be penalized:
- progressively,
- or via a documented threshold.

A score close to 0 corresponds to a reasonable charge,
a negative score to an extreme charge.

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
  → moderate charge, physicochemically plausible peptide.

- **Negative score**
  → excessive charge (positive or negative).

The link between score and biological interpretation must be clear.

---

### Edge cases to handle

- Empty peptide
- Non-standard amino acids
- Length outside the 8–11 aa range

In these cases:
- return a penalty or a documented neutral score,
- never raise an uncontrolled exception.

---

### Limitations to state explicitly in the report (mandatory)

- The real pI **is not calculated**.
- The pH dependence is ignored.
- HLA-specific effects are not taken into account.
- The score is a **qualitative proxy**, not an exact physicochemical measurement.

---

### Common pitfalls (to avoid)

- ❌ Trying to calculate an exact pI
- ❌ Favoring extreme charges
- ❌ Not normalizing by length (if relevant)
- ❌ Producing a score that is difficult to interpret

---

### What this module does NOT claim to do

- It does not predict HLA binding.
- It does not model the real environment of the peptide.
- It does not allow concluding on its own about a neoepitope.

It provides **a complementary signal** to hydrophobicity,
intended to be combined with the other modules.

---

### Critical question you must be able to answer

> *"Why is an extreme charge penalized in your model,
> and how is this choice biologically reasonable despite its simplicity?"*

If you can clearly answer this question,
then your implementation is scientifically consistent.
