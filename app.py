"""
Streamlit Viewer — Audience Segmentation Tool.

Launch with:
    streamlit run app.py

This app provides an interactive interface for exploring segmentation
results.  It can either load pre-computed results from the results/
directory or run the pipeline on-the-fly from the Streamlit UI.

Sections:
  1. Cluster scatter plot (interactive Plotly)
  2. Segment profile cards with key metrics
  3. Company explorer — filter by cluster to see member companies
  4. Outreach Strategy — AI-generated talking points per segment
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Audience Segmentation Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
SEGMENTS_CSV = RESULTS_DIR / "segments.csv"
PROFILES_JSON = RESULTS_DIR / "segment_profiles.json"
PROSPECTS_CSV = PROJECT_ROOT / "data" / "prospects.csv"


# ---------------------------------------------------------------------------
# Data loading (cached so Streamlit does not re-read on every interaction)
# ---------------------------------------------------------------------------
@st.cache_data
def load_segments() -> pd.DataFrame | None:
    """
    Load the segmented prospects CSV.  Returns None if the file does
    not exist yet (pipeline has not been run).
    """
    if SEGMENTS_CSV.exists():
        return pd.read_csv(SEGMENTS_CSV)
    return None


@st.cache_data
def load_profiles() -> dict | None:
    """
    Load the segment profiles JSON.  Returns None if not found.
    """
    if PROFILES_JSON.exists():
        with open(PROFILES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def run_pipeline_from_ui() -> None:
    """
    Run the segmentation pipeline from within the Streamlit app.
    This is triggered by a sidebar button when no pre-computed results
    exist.

    Uses st.spinner to give feedback during the (potentially slow)
    pipeline execution.
    """
    from segment import run_pipeline

    with st.spinner("Running segmentation pipeline — this may take a minute..."):
        run_pipeline(
            input_path=str(PROSPECTS_CSV),
            output_dir=str(RESULTS_DIR),
            n_clusters="auto",
            use_umap=True,
            run_dbscan_comparison=True,
            verbose=False,
        )
    # Clear the cached data so fresh results are loaded
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> str | None:
    """
    Render the sidebar with pipeline controls and segment filter.

    Returns the selected segment name (or None for "All Segments").
    """
    st.sidebar.title("Audience Segmentation")
    st.sidebar.markdown(
        "A portfolio project by **CJ Fleming** demonstrating NLP, "
        "unsupervised learning, and production Python."
    )
    st.sidebar.markdown("---")

    # If results don't exist, offer to run the pipeline
    if not SEGMENTS_CSV.exists():
        st.sidebar.warning("No results found. Run the pipeline first.")
        if st.sidebar.button("Run Pipeline", type="primary"):
            run_pipeline_from_ui()
        return None

    # Segment filter dropdown
    df = load_segments()
    if df is not None and "segment_name" in df.columns:
        segments = ["All Segments"] + sorted(df["segment_name"].dropna().unique().tolist())
        selected = st.sidebar.selectbox("Filter by Segment", segments)
        return selected if selected != "All Segments" else None

    return None


# ---------------------------------------------------------------------------
# Main content sections
# ---------------------------------------------------------------------------


def render_header() -> None:
    """Page header with project title and description."""
    st.title("Programmatic Audience Segmentation")
    st.markdown(
        "Interactive exploration of advertising prospect segments generated "
        "by NLP-driven clustering (TF-IDF + UMAP + K-Means)."
    )
    st.markdown("---")


def render_scatter_plot(df: pd.DataFrame, selected_segment: str | None) -> None:
    """
    Render the interactive Plotly scatter plot of the 2-D projection.

    If a segment is selected in the sidebar, that segment is highlighted
    while others are dimmed.
    """
    st.subheader("Cluster Visualization")

    # We need the pre-computed visualization — if it exists, embed it.
    # Otherwise, build a scatter from the segments CSV (which has
    # cluster_id but not 2-D coordinates).  For the full interactive
    # experience, we reconstruct the projection.
    viz_path = RESULTS_DIR / "visualization.html"

    if viz_path.exists() and selected_segment is None:
        # Show the pre-built HTML directly — fast path
        with open(viz_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=720, scrolling=False)
    else:
        # Build a scatter plot from the data using cluster_id as colour.
        # This path is used when filtering by segment or when the HTML
        # file hasn't been generated.
        if selected_segment:
            # Highlight selected segment, dim others
            plot_df = df.copy()
            plot_df["highlight"] = plot_df["segment_name"].apply(
                lambda s: s if s == selected_segment else "Other"
            )
            color_col = "highlight"
        else:
            color_col = "segment_name"
            plot_df = df

        fig = px.scatter(
            plot_df,
            x="annual_revenue",
            y="current_ad_spend_estimate",
            color=color_col,
            hover_data=["company_name", "industry", "social_presence_score"],
            title="Segments by Revenue vs. Ad Spend",
            labels={
                "annual_revenue": "Annual Revenue ($)",
                "current_ad_spend_estimate": "Estimated Ad Spend ($)",
            },
            template="plotly_white",
            height=600,
        )
        fig.update_traces(
            marker=dict(size=8, opacity=0.75, line=dict(width=0.5, color="white"))
        )
        fig.update_layout(
            legend_title_text="Segment",
            font=dict(family="Inter, Arial, sans-serif", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_profile_cards(profiles: list[dict], selected_segment: str | None) -> None:
    """
    Display a card for each segment showing key metrics, industry mix,
    and campaign themes.
    """
    st.subheader("Segment Profiles")

    # Filter to selected segment if applicable
    if selected_segment:
        profiles = [p for p in profiles if p["segment_name"] == selected_segment]

    # Lay out profiles in a responsive column grid (2 per row)
    for i in range(0, len(profiles), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(profiles):
                break
            p = profiles[idx]
            with col:
                st.markdown(f"### {p['segment_name']}")
                st.markdown(
                    f"**{p['size']}** companies &mdash; "
                    f"**{p['percentage']}%** of total"
                )

                # Key metrics in a compact metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Med. Revenue", f"${p['median_revenue']:,.0f}")
                m2.metric("Med. Employees", f"{p['median_employees']:,.0f}")
                m3.metric("Med. Ad Spend", f"${p['median_ad_spend']:,.0f}")
                m4.metric("Social Score", f"{p['median_social_score']:.0f}")

                # Industry distribution
                with st.expander("Industry Mix"):
                    for industry, pct in p["industry_distribution"].items():
                        st.markdown(f"- **{industry}**: {pct}%")

                # Campaign themes
                if p.get("campaign_themes"):
                    with st.expander("Campaign Themes"):
                        for theme in p["campaign_themes"]:
                            st.markdown(f"- {theme}")

                st.markdown("---")


def render_company_explorer(df: pd.DataFrame, selected_segment: str | None) -> None:
    """
    Filterable table of companies, optionally narrowed to a single
    segment.  Shows the most relevant columns for a sales team.
    """
    st.subheader("Company Explorer")

    display_df = df.copy()
    if selected_segment:
        display_df = display_df[display_df["segment_name"] == selected_segment]

    # Select and reorder columns for readability
    display_cols = [
        "company_name", "segment_name", "industry", "annual_revenue",
        "employee_count", "current_ad_spend_estimate",
        "social_presence_score", "target_demographics",
    ]
    # Only include columns that actually exist in the dataframe
    display_cols = [c for c in display_cols if c in display_df.columns]
    display_df = display_df[display_cols]

    # Format currency columns for readability
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        column_config={
            "company_name": st.column_config.TextColumn("Company"),
            "segment_name": st.column_config.TextColumn("Segment"),
            "industry": st.column_config.TextColumn("Industry"),
            "annual_revenue": st.column_config.NumberColumn(
                "Revenue", format="$%d"
            ),
            "employee_count": st.column_config.NumberColumn("Employees"),
            "current_ad_spend_estimate": st.column_config.NumberColumn(
                "Ad Spend", format="$%d"
            ),
            "social_presence_score": st.column_config.NumberColumn(
                "Social Score", format="%d / 100"
            ),
            "target_demographics": st.column_config.TextColumn("Target Demo"),
        },
    )
    st.caption(f"Showing {len(display_df)} companies")


def render_outreach_strategies(
    profiles: list[dict], selected_segment: str | None
) -> None:
    """
    Display outreach strategy recommendations per segment.

    These are rule-based talking points generated by the pipeline,
    designed to give a sales team immediate, actionable guidance for
    each audience segment.
    """
    st.subheader("Outreach Strategies")
    st.markdown(
        "Actionable talking points and channel recommendations for each "
        "segment, based on revenue tier and engagement profile."
    )

    if selected_segment:
        profiles = [p for p in profiles if p["segment_name"] == selected_segment]

    for p in profiles:
        with st.expander(f"{p['segment_name']} — Outreach Strategy", expanded=True):
            st.markdown(f"**Segment size:** {p['size']} companies")
            st.markdown(f"**Median revenue:** ${p['median_revenue']:,.0f}")
            st.markdown(f"**Median ad spend:** ${p['median_ad_spend']:,.0f}")
            st.markdown("---")
            st.markdown(p["outreach_strategy"])

            # Top demographics for context
            if p.get("top_demographics"):
                st.markdown("**Key demographics to target:**")
                for demo in p["top_demographics"]:
                    st.markdown(f"- {demo}")


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Compose all sections into the Streamlit page."""
    render_header()
    selected_segment = render_sidebar()

    # Load data — if not available, show instructions
    df = load_segments()
    raw_profiles = load_profiles()

    if df is None or raw_profiles is None:
        st.info(
            "No segmentation results found. Click **Run Pipeline** in the "
            "sidebar, or run the CLI first:\n\n"
            "```bash\n"
            "python segment.py --input data/prospects.csv --clusters auto "
            "--output results/\n"
            "```"
        )
        return

    # Extract the K-Means profiles (primary) from the JSON structure.
    # The JSON can be either a flat list (older format) or a dict keyed
    # by algorithm name.
    if isinstance(raw_profiles, dict):
        profiles = raw_profiles.get("kmeans", [])
    elif isinstance(raw_profiles, list):
        profiles = raw_profiles
    else:
        profiles = []

    if not profiles:
        st.error("Profiles JSON is empty or malformed.")
        return

    # Render all sections
    render_scatter_plot(df, selected_segment)
    render_profile_cards(profiles, selected_segment)
    render_company_explorer(df, selected_segment)
    render_outreach_strategies(profiles, selected_segment)

    # Footer
    st.markdown("---")
    st.markdown(
        "*Built by CJ Fleming — demonstrating NLP, unsupervised learning, "
        "and production Python. "
        "[View on GitHub](https://github.com/cjfleming)*"
    )


if __name__ == "__main__":
    main()
