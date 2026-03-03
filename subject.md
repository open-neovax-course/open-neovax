# BIOINFO L3 PROJECT — OPEN-NEOVAX

## Rational Design of mRNA Cancer Vaccines using Explainable Scoring

---

## 0. General objective

The goal of this project is to build **Open-NeoVax**, a **modular, interpretable and reproducible** Python framework to **prioritize HLA class I (CD8+) neo-epitopes** from tumor mutations.

The goal is **not** to compete with industrial or academic tools like NetMHCpan, but to:

- understand **the biological reasoning** behind neo-epitope prioritization,
- implement **explicit and traceable proxies**,
- work within a **professional software framework** (Git, tests, style, reproducibility).

### Final deliverables

- A functional Open-NeoVax tool (provided + student modules)
- One Python module per group
- One PDF report per group
- A short oral presentation during the final session

---

## 1. Scientific context (refresher)

Cancer is associated with genomic alterations (somatic mutations) that can modify protein sequences.
These mutated proteins are degraded into peptides, a fraction of which are presented on the cell surface by **HLA class I** molecules.

Some mutated peptides, called **neo-epitopes**, can be recognized as non-self by CD8+ T lymphocytes and represent potential targets for therapeutic vaccines.

Identifying these neo-epitopes relies on several biological constraints:

- HLA compatibility,
- plausible cellular presentation,
- difference from self,
- physicochemical trade-offs.

In this project, these constraints will be modeled via **explicit scoring modules**.

---

## 2. Project scope

### 2.1 Hypotheses and limitations

- HLA class I only (CD8+)
- Peptide length **8 to 11 amino acids**
- Multiple HLA alleles possible (e.g. A*02:01, B*07:02...)
- TCR immunogenicity is **not directly predicted**
- Scores are **biological proxies**, not clinical predictions

### 2.2 Input data

The main pipeline works from a **list of already-generated peptides**.

Main format: CSV

Required columns:
- `candidate_id`
- `peptide_wt`
- `peptide_mut`
- `mut_pos_1based`

Optional columns:
- `gene`
- `hla_allele`
- `note`

> Simplified VCF files may be provided for illustration, but their full parsing is **not required** for project validation.

---

## 3. Technical environment

- Python >= 3.9
- Virtual environment via `venv`
- Dependency installation with `pip`
- Supported systems: Linux, macOS, Windows

### 3.1 Getting started

```bash
# 1. Clone the repository
git clone <REPO_URL>
cd open-neovax

# 2. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify everything works
pytest tests/ -v             # tests
ruff check .                 # lint
black --check .              # format

# 5. Launch the interface
streamlit run app.py
```

---

## 4. Code quality (mandatory)

All student modules must comply with:

- **PEP8** (priority)
- `black` (formatting)
- `ruff` (lint)
- Type hints on public functions
- Explicit constants (no magic numbers)
- Docstring explaining hypothesis, inputs, outputs and limitations
- Explicit error handling
- Unit tests (>= 3 per module)

---

## 5. Software architecture (provided)

```
open-neovax/
├── app.py                  # Streamlit interface
├── logic/
│   ├── types.py            # Candidate dataclass
│   ├── data_loader.py      # CSV loading
│   ├── orchestrator.py     # Module discovery and execution
│   └── scoring.py          # Normalization and aggregation
├── modules/
│   ├── template_module.py  # Template to copy
│   └── <groupe_XX.py>      # One file = one group
├── data/
│   ├── patient_zero.csv    # Test data (18 candidates)
│   ├── hla_pssm_A0201.csv  # PSSM matrix HLA-A*02:01
│   └── human_peptides_small.txt  # Self corpus (500 peptides)
└── tests/
    ├── test_core.py        # Core tests (provided)
    ├── test_template.py    # Template tests (provided)
    └── test_groupe_XX.py   # YOUR tests (to create)
```

Students **only modify their module** and their tests.

---

## 6. Group organization

- Groups of **2 to 3 students**
- One module per group
- Modules assigned by the instructor

---

## 7. Departments & module examples

### Department A — Peptide properties

Hydrophobicity, charge, complexity, delta WT/MUT.

### Department B — Processing / presentation

Proteasome, TAP, ERAP, sanity checks.

### Department C — HLA

Anchors P2/P9, secondary anchors, delta binding, multi-allele (bonus).

### Department D — Safety / self

Self-similarity, mutation-in-window, viral similarity (bonus).

---

## 8. Student module API

```python
def get_score(candidate: Candidate) -> tuple[str, float]:
    ...
```

---

## 9. Git & collaboration

### Workflow

1. **Clone** the repository (`git clone`)
2. **Create a branch** (`git checkout -b groupe-XX`)
3. **Work** on your module and tests
4. **Commit** regularly with clear messages (convention `feat:`, `fix:`, `test:`)
5. **Push** your branch (`git push -u origin groupe-XX`)
6. **Create a Pull Request** to `main` on GitHub
7. **Check CI**: all 3 checks (ruff, black, pytest) must pass green
8. **Respond to reviews** from the instructor, fix and re-push if needed

### Rules

- **Never push to `main`** — always use a branch + PR
- **Never modify another group's file**
- **Never modify core files** (`logic/`, `app.py`)
- A PR template fills in automatically: complete all sections
- See `CONTRIBUTING.md` for the detailed guide and common mistakes

---

## 10. Session schedule (6 x 4h)

1. Onboarding & Git
2. Naive score
3. External data
4. Robustness
5. Integration
6. Analysis & presentations

---

## 11. Research component

In addition to the scoring module, each group must conduct a **critical analysis** of their approach.

### Expectations

- Analyze the **strengths and limitations** of your module within the pipeline context
- Identify **cases where your score conflicts** with other modules
- Propose **improvements** or **extensions**
- Answer the **deep-dive questions** (see `deep-dive-questions.md`)
- Cite at least **1 relevant scientific reference** (article, database, or bioinformatics tool)

### Format

- 1 to 2 pages integrated into the PDF report
- This component accounts for **10% of the final grade**

---

## 12. Deliverables & grading

| Criterion             | Weight |
|-----------------------|--------|
| Code                  | 40%    |
| Science / Report      | 20%    |
| Research / Analysis   | 10%    |
| Integration           | 10%    |
| Collaboration         | 20%    |

---

## 13. Possible extensions

Advanced VCF, extended multi-allele, benchmarks, Docker, visualization.

---

Project designed to reproduce a **realistic modern bioinformatics setting**, combining scientific rigor and software best practices.
