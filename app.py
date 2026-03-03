"""Open-NeoVax — Streamlit interface for neo-epitope prioritization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from logic.data_loader import load_candidates
from logic.orchestrator import run_modules
from logic.scoring import aggregate
from logic.types import Candidate

DATA_DIR = Path(__file__).resolve().parent / "data"
MODULES_DIR = Path(__file__).resolve().parent / "modules"


# ── Utility functions ───────────────────────────────────────────────


def candidates_to_dataframe(candidates: list[Candidate]) -> pd.DataFrame:
    """Convert a list of candidates into a DataFrame for display."""
    rows = []
    for c in candidates:
        row = {
            "ID": c.candidate_id,
            "Peptide WT": c.peptide_wt,
            "Peptide MUT": c.peptide_mut,
            "Mutation pos.": c.mut_pos_1based,
            "Gene": c.gene,
            "HLA": c.hla_allele,
        }
        for score_name, score_value in sorted(c.scores.items()):
            row[score_name] = round(score_value, 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Interface ───────────────────────────────────────────────────────

st.set_page_config(page_title="Open-NeoVax", layout="wide")
st.title("Open-NeoVax")
st.markdown("Modular pipeline for HLA-I neo-epitope prioritization")

# ── Data loading ────────────────────────────────────────────────────

st.header("1. Load candidates")

use_demo = st.checkbox("Use demo file (patient_zero.csv)", value=True)

csv_path: Path | None = None

if use_demo:
    demo_path = DATA_DIR / "patient_zero.csv"
    if demo_path.exists():
        csv_path = demo_path
        st.info(f"Demo file: `{demo_path.name}`")
    else:
        st.error("patient_zero.csv not found in data/.")
else:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        tmp_path = DATA_DIR / "_uploaded.csv"
        tmp_path.write_bytes(uploaded.read())
        csv_path = tmp_path

# ── Run analysis ────────────────────────────────────────────────────

st.header("2. Analysis")

if st.button("Run analysis", disabled=(csv_path is None)):
    if csv_path is None:
        st.warning("Please load a CSV file first.")
    else:
        try:
            candidates = load_candidates(csv_path)
            st.success(f"{len(candidates)} candidates loaded.")
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"Loading error: {exc}")
            st.stop()

        with st.spinner("Running scoring modules..."):
            candidates = run_modules(candidates, MODULES_DIR)

        candidates = aggregate(candidates)

        st.session_state["candidates"] = candidates
        st.session_state["has_results"] = True

# ── Display results ─────────────────────────────────────────────────

if st.session_state.get("has_results"):
    results: list[Candidate] = st.session_state["candidates"]

    st.header("3. Results")

    df = candidates_to_dataframe(results)
    st.dataframe(df, use_container_width=True)

    st.header("4. Candidate details")

    options = [c.candidate_id for c in results]
    selected_id = st.selectbox("Select a candidate", options)

    selected = next(c for c in results if c.candidate_id == selected_id)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Peptide WT**: `{selected.peptide_wt}`")
        st.markdown(f"**Peptide MUT**: `{selected.peptide_mut}`")
        st.markdown(f"**Mutation position**: {selected.mut_pos_1based}")
        if selected.gene:
            st.markdown(f"**Gene**: {selected.gene}")
        if selected.hla_allele:
            st.markdown(f"**HLA**: {selected.hla_allele}")

    with col2:
        individual_scores = {
            k: v for k, v in selected.scores.items() if k != "total_score"
        }
        if individual_scores:
            scores_df = pd.DataFrame(
                {
                    "Score": list(individual_scores.keys()),
                    "Value": list(individual_scores.values()),
                }
            )
            st.bar_chart(scores_df.set_index("Score"))
        else:
            st.info("No individual scores available.")

        if "total_score" in selected.scores:
            st.metric("Total score", f"{selected.scores['total_score']:.4f}")
