# Report guidelines

## Format

- **PDF only** — named `rapport_groupe_XX.pdf`
- **Maximum 10 pages** (no minimum — quality over quantity)
- **Language**: English or French
- **Deadline: May 1st**
- **Submit by email** to `philippe.rinaudo [at] universite-paris-saclay [dot] fr`
- Structure is free, but the report must cover the sections below

## What to include

### 1. Module description

- Which module(s) did you implement? Which department?
- What is the biological hypothesis behind your score?
- How is the score computed? (algorithm overview, not a code dump)
- What data does it use? (PSSM, self corpus, lookup tables, etc.)

### 2. Implementation details

- How do you handle edge cases? (empty peptide, invalid AA, missing data file)
- How is the data loaded? (caching strategy, fallback behavior)
- Key design decisions and why you made them

### 3. Results

- Show 2-3 candidates that your module scores well and 2-3 that it scores poorly
- Are these results biologically coherent? Explain why
- If you implemented multiple modules: how do they complement each other?

### 4. Critical analysis

- Identify at least one **false positive**: a candidate your module ranks high but that is actually a bad candidate (and explain why)
- Identify at least one **limitation** of your approach
- Suggest at least one **improvement** you would make with more time

### 5. ML global scoring (if applicable)

If your group worked on the ML project (issue #39):

- Which models did you try? What were the cross-validation results?
- **Feature importance**: which modules are the most predictive? Why does this make biological sense?
- **Final ranking**: does CAND_01 (GOLD) end up at the top?
- If you used `patient_real.csv`: does the model generalize to real experimental data?

### 6. Deep-dive questions (optional)

You may answer some or all of the deep-dive questions from `docs/deep-dive-questions.md`:
- 5 common questions (applicable to all groups)
- 3 department-specific questions

These are optional but can strengthen your critical analysis section.

### 7. References

- At least **1 scientific reference** related to your module's biological basis
- Additional references for methods, data sources, or tools used

## Tips

- Focus on **why**, not **what**. Don't paste your code — explain your reasoning
- Use figures: a well-chosen plot (score distribution, feature importance, ranking) is worth more than a paragraph
- Be honest about limitations — this is valued more than pretending everything works perfectly
- If something didn't work as expected, explain what you learned from it
