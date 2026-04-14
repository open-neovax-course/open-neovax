# Open-NeoVax

Modular, interpretable and reproducible Python framework for **HLA class I (CD8+) neoepitope prioritization** from tumor mutations.

Built as a hands-on bioinformatics project: students implement individual scoring modules that plug into a shared pipeline.

## Quick start

```bash
# Clone and setup
git clone https://github.com/open-neovax-course/open-neovax.git
cd open-neovax
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify
pytest tests/ -v
ruff check .
black --check .

# Launch
streamlit run app.py
```

## How it works

The pipeline scores candidate neoepitopes through independent modules, then ranks them:

```
CSV input → Load candidates → Run scoring modules → Normalize → Aggregate → Ranked output
```

Each module implements a single biological proxy (hydrophobicity, HLA anchor strength, self-similarity, etc.) and exposes one function:

```python
def get_score(candidate: Candidate) -> tuple[str, float]:
    return ("my_score_name", 0.85)
```

Modules are discovered automatically — drop a `.py` file in `modules/` and it runs.

## Project structure

```
open-neovax/
├── app.py                     # Streamlit interface
├── logic/
│   ├── types.py               # Candidate dataclass
│   ├── data_loader.py         # CSV loading + validation
│   ├── orchestrator.py        # Module discovery + safe execution
│   └── scoring.py             # Normalization + weighted aggregation
├── modules/
│   ├── template_module.py     # Template to copy for your module
│   └── groupe_XX.py           # Your module (one per group)
├── data/
│   ├── patient_zero.csv       # Test dataset (18 candidates)
│   ├── hla_pssm_A0201.csv    # PSSM matrix for HLA-A*02:01
│   └── human_peptides_small.txt  # Self peptide corpus (500 entries)
├── tests/
│   ├── test_core.py           # Core pipeline tests
│   ├── test_template.py       # Template module tests
│   └── test_groupe_XX.py     # Your tests (>= 3)
└── docs/                      # Project documentation
    ├── subject.md             # Full project description
    ├── common-rules.md        # Rules for all modules
    ├── deep-dive-questions.md # Report questions
    ├── departments-and-modules.md  # Module index
    ├── departments/           # Department briefs (A, B, C, D)
    └── modules/               # Detailed specs per module
```

## For students

1. Read the [subject](docs/subject.md), the [common rules](docs/common-rules.md), and your [module spec](docs/departments-and-modules.md)
2. Copy `modules/template_module.py` to `modules/groupe_XX.py`
3. Create your branch: `git checkout -b groupe-XX`
4. Implement your `get_score` function
5. Write tests in `tests/test_groupe_XX.py` (minimum 3)
6. Push and open a Pull Request to `main`
7. CI must pass (ruff + black + pytest) before review

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full Git workflow.

## Scoring modules

| Dept | Focus | Modules |
|------|-------|---------|
| **A** | Physicochemical properties | Hydrophobicity, charge, complexity, delta WT/MUT |
| **B** | Processing & presentation | Proteasome, TAP, ERAP, sanity checks |
| **C** | HLA binding | Anchor P2, anchor P9, secondary anchors, delta binding |
| **D** | Safety & self | Exact self-match, approximate self-match, mutation in window |

## Requirements

- Python >= 3.9
- Dependencies: `streamlit`, `pandas`, `pytest`, `ruff`, `black`

## Contact

Philippe Rinaudo — `philippe.rinaudo [at] universite-paris-saclay [dot] fr`
