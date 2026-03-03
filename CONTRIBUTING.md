# Contributing guide — Open-NeoVax

This guide walks you through contributing to the project step by step.
Follow each step in order.

---

## Step 1: Clone the repository

```bash
git clone <REPO_URL>
cd open-neovax
```

## Step 2: Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows
```

## Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

Verify everything works:

```bash
pytest tests/ -v       # tests
ruff check .           # lint
black --check .        # format
```

## Step 4: Create your branch

**Never work directly on `main`.**

```bash
git checkout -b groupe-XX
```

Replace `XX` with your group number (e.g. `groupe-03`).

## Step 5: Work on your module

1. Copy the template:
   ```bash
   cp modules/template_module.py modules/groupe_XX.py
   ```
2. Edit `modules/groupe_XX.py` with your scoring logic
3. Create your tests in `tests/test_groupe_XX.py`

## Step 6: Commit regularly

Make **small and frequent** commits with clear messages:

```bash
git add modules/groupe_XX.py tests/test_groupe_XX.py
git commit -m "feat: Kyte-Doolittle hydrophobicity score"
```

### Commit message convention

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

## Step 7: Push your branch

```bash
git push -u origin groupe-XX
```

## Step 8: Create a Pull Request

1. Go to GitHub in the repository
2. Click **"Compare & pull request"**
3. Fill in the PR template (it appears automatically)
4. Click **"Create pull request"**

## Step 9: Check CI

After creating your PR, CI (continuous integration) runs automatically.
Verify that all 3 checks pass:

- **ruff**: lint (code style)
- **black**: formatting
- **pytest**: unit tests

If a check fails, fix locally, commit and push again.
CI will rerun automatically.

## Step 10: Respond to reviews

The instructor may leave comments on your PR.
Read them carefully, fix, commit and push.

---

## Common mistakes

### I forgot to create my branch

If you started working on `main`:

```bash
git checkout -b groupe-XX
# Your changes are now on the new branch
```

### I committed on main by mistake

```bash
# Create the branch from current state
git branch groupe-XX
# Go back to a clean main
git checkout main
git reset --hard origin/main
# Switch to your branch
git checkout groupe-XX
```

### CI tests are failing

```bash
# Run the same commands locally:
ruff check .
black --check .
pytest tests/ -v

# To auto-fix formatting:
black .
```

### Merge conflict

```bash
git checkout main
git pull
git checkout groupe-XX
git merge main
# Resolve conflicts in your editor, then:
git add .
git commit -m "fix: resolve merge conflict"
```

### `__pycache__` in my commit

This directory should **never** be committed (it is in `.gitignore`).
If you added it by mistake:

```bash
git rm -r --cached __pycache__
git commit -m "fix: remove __pycache__ from git tracking"
```

---

## Fundamental rules

1. **Never push to `main`** — always use a branch + PR
2. **Never modify another group's file**
3. **Never modify core files** (`logic/`, `app.py`)
4. **Always check CI** before requesting a review
