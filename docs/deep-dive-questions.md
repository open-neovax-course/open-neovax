# Deep-dive questions

This document lists the questions each group must answer in their report.
There are **mandatory questions** (common to all) and **department-specific questions**.

**Format**: 1 to 2 pages integrated into the PDF report.

---

## Mandatory questions (all groups)

These 5 questions are the same for all groups, regardless of module.

### Q1 — Validation

How did you verify that your score is biologically coherent?
Give a concrete example of a well-scored candidate and a poorly-scored candidate.
Explain why these results are expected.

### Q2 — Pipeline contribution

What is the impact of your module on the final candidate ranking?
Is it dominant, complementary, or negligible?
What happens if its weight is set to 0 in the total score?

### Q3 — Inter-module conflict

Give a concrete example of a candidate where your score conflicts with another department's score (HLA, self-similarity, physicochemical properties, etc.).
How should the pipeline handle this conflict?

### Q4 — False positives and false negatives

Describe a typical false positive of your module (candidate wrongly favored) and a false negative (candidate wrongly penalized).
What are the biological consequences of each?

### Q5 — Outlook

Propose an improvement or extension of your module.
Cite at least **1 relevant scientific reference** (article, database, or bioinformatics tool).

---

## Department-specific questions

In addition to the 5 mandatory questions, answer the 3 questions for your department.

### Department A — Physicochemical peptide properties

**A-Q1**: Is there a candidate ranked highly overall (good HLA score, good self score) but penalized by your physicochemical score? Why?

**A-Q2**: Does your score indirectly favor certain HLA anchor amino acids? For example, is leucine (good HLA anchor) also favored by your physicochemical scale?

**A-Q3**: Compare the wild-type and mutated peptides for 3 candidates from the test dataset. Does the mutation systematically improve your score? In which cases does it degrade it?

### Department B — Processing & cellular presentation

**B-Q1**: Does your processing module (proteasome, TAP or ERAP) capture a signal that is truly independent from HLA binding, or is there a correlation between good presentation and good binding?

**B-Q2**: What is the impact of a peptide containing invalid characters (X, *, etc.) on the overall pipeline? How does your module detect it and what does it return?

**B-Q3**: If the dataset contained 10 or 11-mer peptides instead of 9-mers, would your module still work? What adaptations would be needed?

### Department C — HLA binding

**C-Q1**: Can a peptide with good P2/P9 anchors but penalized by the self-similarity module (D1 or D2) still rank in the top 5? Under what conditions?

**C-Q2**: Is your anchor score sufficient alone to correctly rank candidates, or does it need to be combined with other modules? Justify with examples from the test dataset.

**C-Q3**: The provided PSSM matrix (`hla_pssm_A0201.csv`) is specific to HLA-A*02:01. How would you adapt your module to support another allele (e.g. HLA-B*07:02)? What data would be needed?

### Department D — Safety & self-similarity

**D-Q1**: Are there peptides highly scored by HLA and physicochemical modules that your self-similarity module "eliminates" from the top? Give a concrete example.

**D-Q2**: What happens if you slightly relax your penalty threshold (e.g. Hamming distance of 1 instead of 0 for exact match)? Does the ranking change significantly?

**D-Q3**: Module D3 (mutation-in-window) detects cases where the mutation position is outside the peptide window. Biologically, why is this case problematic for a neo-epitope vaccine?
