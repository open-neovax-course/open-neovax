# Presentation guidelines

## Format

- **8-10 minutes** per group + 3-5 minutes of questions
- **All group members must speak**
- Support is free: slides, live demo, whiteboard, Jupyter notebook — your choice
- **Deadline: May 1st** (presentation day)

## Structure

Your presentation should cover the following, in any order you prefer:

### 1. Module (2 min)

- Which module, which department
- The biological hypothesis **in one sentence**
- Example: "We score how well the peptide's anchor positions match HLA-A*02:01 preferences"

### 2. Implementation (3 min)

- How the score is computed (simplified algorithm, not code)
- What data is used (PSSM, self corpus, etc.)
- How edge cases are handled (don't crash, return neutral score)
- One design decision you're proud of

### 3. Results (2 min)

- Show 2-3 candidates: one well-scored, one poorly-scored
- Is this biologically coherent?
- Live demo with Streamlit is welcome but optional

### 4. Critical analysis (2 min)

- One **concrete false positive** from patient_zero (a candidate your module ranks high but shouldn't be)
- One **limitation** of your approach
- One **improvement** you would make

### 5. ML project (1 min, if applicable)

If your group worked on the global scoring model:
- Which models did you try?
- Feature importance: is your module among the top predictors?
- Final ranking: where does CAND_01 end up?

## What I will ask during Q&A

### Technical questions
- "What happens if the peptide contains an X?"
- "How many times is the PSSM file loaded?"
- "Why `.get(aa, 0.0)` instead of `[aa]`?"

### Biological questions
- "Is a high score from your module enough for a good vaccine candidate?"
- "Which other module complements yours best?"
- "Would this work with a different HLA allele?"

### The key question
> "Name a candidate that your module ranks incorrectly, and explain why."

If you can answer this clearly, you understand your module.

## Evaluation criteria

| Criteria | What I look for |
|----------|----------------|
| Biological hypothesis | Clear, specific, and justified |
| Code explanation | Understandable without reading the code |
| Results shown | Concrete examples, not just "it works" |
| False positive identified | A specific candidate with explanation |
| Q&A answers | Demonstrates understanding, not memorization |
| All members participate | Everyone speaks and can answer questions |
