# Rules common to ALL modules

This section applies **to all groups, all departments and all modules**.
It defines the **minimum technical and scientific contract** that each module must respect
to be properly integrated into Open-NeoVax.

---

## Required signature (module API)

Each student module must expose **a single public function**:

```python
def get_score(candidate: Candidate) -> tuple[str, float]:
    ...
```

This function is the module's official API.

The Open-NeoVax core:

- dynamically imports your Python file,
- calls only `get_score(candidate)`,
- knows nothing else about your implementation.

### What this means

- You can define as many internal functions as needed in your file
  (`compute_score`, `normalize`, `_load_data`, etc.).
- No other function should be called from outside the module.
- `get_score` is the only authorized entry point.

> **Mental rule:**
> Your module is a black box with a single button: `get_score`.

---

## Expected return value

The `get_score` function must return exactly:

```python
(score_name: str, score_value: float)
```

### 1. The score name (`score_name`) must be unique

`score_name` is the key under which your score is stored in
`candidate.scores` (a dictionary).

Two modules returning the same `score_name` would silently overwrite
the previous score.

**Naming convention (mandatory):**

The score name must include the department and the main biological concept.

Format: `<department>_<concept>[_detail]`

Valid examples:
- `A_hydrophobicity_kd`
- `B_proteasome_cterm`
- `C_hla_anchor_p2`
- `D_self_exact_match`

Invalid examples:
- `score`
- `hydrophobicity`
- `hla_score`

### 2. The returned score must be a `float`

- The score must always be a real number (Python `float`).
- `NaN`, `inf`, `-inf` values are **forbidden**.
- Scores can be positive or negative.
- Final normalization is handled by the core (you don't need to do it).
- If a score cannot be computed:
  - either return a documented neutral score,
  - or an explicit, biologically consistent penalty.

### 3. The module must never crash the pipeline

- All errors must be handled within the module.
- An invalid input must never cause an unhandled exception.
- Examples of cases to handle explicitly:
  - invalid amino acids,
  - length outside 8–11 aa,
  - missing data,
  - missing or malformed auxiliary file.

> A failing module must fail **gracefully**, without blocking the overall analysis.

---

## What your module MUST do

### Implement a clear biological hypothesis

Your module corresponds to a specific biological idea.
This hypothesis must be stated in:
- the module docstring,
- the written report.

### Return an interpretable score

The score must have biological meaning.
You must be able to explain:
- what a high score means,
- what a low score means,
- a typical false positive.

### Handle edge cases

Unexpected inputs are normal in bioinformatics.
Your code must be robust and predictable.

### Be documented and tested

Each module must contain:
- a **docstring** describing: the hypothesis, the computation logic, the limitations,
- **unit tests** covering: a nominal case, at least one edge case, an invalid case.

---

## What your module MUST NOT do

- **Load network data** — no Internet access, no external API, no dynamic dependency
- **Modify other scores** — you must never read or write to `candidate.scores` outside your own score
- **Depend on another student module** — no imports from another student module, each module must be independent
- **Use uncontrolled randomness** — no `random()` without a fixed seed, no non-deterministic behavior. Two identical runs must produce the same scores

---

## Using auxiliary files (allowed, but regulated)

Modules may use local data files:
- allowed formats: `.csv`, `.json`, `.txt`
- files included in the Git repository
- loaded locally only

Associated requirements:
- document the file's role in the report,
- handle its absence or bad format gracefully,
- do not assume an absolute path.

---

## Git conventions

### Branch naming

Each group works on its own branch:

```
groupe-XX
```

Replace `XX` with your group number (e.g. `groupe-03`).

### Commit messages

Use the following convention:

| Prefix   | Usage                          |
|----------|--------------------------------|
| `feat:`  | New feature                    |
| `fix:`   | Bug fix                        |
| `test:`  | Add or modify tests            |
| `docs:`  | Documentation                  |
| `style:` | Formatting (black, ruff)       |

Examples:
- `feat: PSSM score computation for P2`
- `fix: handle peptide length < 9`
- `test: add edge case for empty peptide`

### Pull Requests

- All contributions go through a **Pull Request** (PR) to `main`.
- The PR template fills in automatically: complete all sections.
- Never merge yourself: the instructor validates and merges.

---

## Continuous integration (CI)

### Automated pipeline

Each PR automatically triggers 3 checks:

1. **ruff** — lint (detects style errors and common bugs)
2. **black** — formatting (verifies uniform code formatting)
3. **pytest** — unit tests (verifies all tests pass)

### Rules

- All 3 checks must pass **green** for the PR to be accepted.
- If a check fails, fix locally, commit and push: CI will rerun.

### Local verification (before pushing)

```bash
ruff check .           # lint
black --check .        # format (check only)
black .                # format (auto-fix)
pytest tests/ -v       # tests
```

---

## Unit tests

### Minimum requirements

Each student module must have **at least 3 tests** in `tests/test_<module>.py`:

1. **Nominal test** — a standard case that works
2. **Edge-case test** — short peptide, unusual characters, missing data
3. **Invalid-input test** — absurd input, missing file

### File organization

```
tests/
├── __init__.py
├── test_core.py            # core tests (provided)
├── test_template.py        # template tests (provided)
└── test_groupe_XX.py       # YOUR tests (to create)
```

### Minimal example

```python
from logic.types import Candidate
from modules.groupe_XX import get_score

def _make_candidate(peptide_mut="SLMAFTIAV"):
    return Candidate(
        candidate_id="TEST",
        peptide_wt="AAAA",
        peptide_mut=peptide_mut,
        mut_pos_1based=5,
    )

def test_nominal():
    name, value = get_score(_make_candidate())
    assert isinstance(name, str)
    assert isinstance(value, float)

def test_short_peptide():
    name, value = get_score(_make_candidate("AA"))
    assert isinstance(value, float)

def test_empty_peptide():
    name, value = get_score(_make_candidate(""))
    assert isinstance(value, float)
```

---

## Final rule (remember this)

If your module works on its own, cleanly, deterministically,
and can be explained biologically in 5 minutes,
then it is compliant.
