# OPEN-NEOVAX — Department and module index

This document is a **navigation index** to the detailed instructions for each department and module.

Each group must read:
- the **common rules** file (`common-rules.md`),
- the file for **their department**,
- the file for **their module**,
- the **deep-dive questions** file (`deep-dive-questions.md`).

---

## Common rules

- `common-rules.md` — Technical and scientific contract common to all modules.
- `deep-dive-questions.md` — Deep-dive questions for the report (mandatory + department-specific).

---

## Department A — Physicochemical peptide properties

- `department-A.md` — Scientific objective and general logic of the department.

| Module | File | Description |
|--------|------|-------------|
| A1 | `module-A1.md` | Hydrophobicity (Kyte-Doolittle) |
| A2 | `module-A2.md` | Net charge / pI (proxy) |
| A3 | `module-A3.md` | Sequence complexity / entropy |
| A4 | `module-A4.md` | Delta WT vs MUT (mutation impact) |

---

## Department B — Processing & cellular presentation

- `department-B.md` — Scientific objective and general logic of the department.

| Module | File | Description |
|--------|------|-------------|
| B1 | `module-B1.md` | Proteasome (C-terminal proxy) |
| B2 | `module-B2.md` | TAP (transport proxy) |
| B3 | `module-B3.md` | ERAP trimming (N-terminal proxy) |
| B4 | `module-B4.md` | Sanity checks (minimum biological validity) |

---

## Department C — HLA (project core)

- `department-C.md` — Scientific objective and general logic of the department.

| Module | File | Description |
|--------|------|-------------|
| C1 | `module-C1.md` | Anchor position P2 (HLA-I PSSM) |
| C2 | `module-C2.md` | C-terminal anchor (P9 / PΩ) |
| C3 | `module-C3.md` | Secondary anchors (positional contributions) |
| C4 | `module-C4.md` | Delta binding WT vs MUT |

---

## Department D — Safety & self-similarity

- `department-D.md` — Scientific objective and general logic of the department.

| Module | File | Description |
|--------|------|-------------|
| D1 | `module-D1.md` | Exact self-similarity |
| D2 | `module-D2.md` | Approximate self-similarity (option) |
| D3 | `module-D3.md` | Mutation in window |

---

## What is evaluated for ALL modules

- Clear and justified hypothesis
- Readable and robust code
- Relevant unit tests
- Ability to explain:
  - what the module does
  - what it cannot do
  - a typical false positive

---

End of document.
