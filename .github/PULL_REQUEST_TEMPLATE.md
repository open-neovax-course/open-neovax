## Group information

- **Group**: <!-- e.g. Group 01 -->
- **Members**: <!-- First Last, First Last -->
- **Module**: <!-- e.g. A1 — Hydrophobicity -->
- **Score name**: <!-- e.g. A_hydrophobicity_kd -->

## Description

<!-- Briefly describe what your module does and the underlying biological hypothesis. -->

## Changes

<!-- List added or modified files. -->

- [ ] `modules/<your_module>.py`
- [ ] `tests/test_<your_module>.py`
- [ ] Data file(s) (if applicable): <!-- e.g. data/my_file.csv -->

## Checklist

### Code
- [ ] `get_score(candidate)` is the only public function
- [ ] The return value is a `tuple[str, float]`
- [ ] The score name follows the convention `<dept>_<concept>` (e.g. `A_hydrophobicity_kd`)
- [ ] No `NaN`, `inf` or `-inf` in returned values
- [ ] Edge cases are handled (invalid peptide, missing data)
- [ ] Docstring present: hypothesis, inputs, outputs, limitations

### Tests
- [ ] At least **3 unit tests** in `tests/test_<module>.py`
- [ ] One nominal test (standard case)
- [ ] One edge-case test
- [ ] One invalid-input test

### Quality
- [ ] `ruff check .` passes
- [ ] `black --check .` passes
- [ ] `pytest tests/ -v` passes
- [ ] No magic numbers (named constants used)
- [ ] No network access or external API calls

### Independence
- [ ] My module does not import any other student module
- [ ] My module does not modify `candidate.scores` directly
- [ ] My module does not depend on execution order
