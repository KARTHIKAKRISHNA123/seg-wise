# =============================================================================
# app.py — SegWise Streamlit Dashboard
#
# PURPOSE:
#   Interactive web dashboard that loads the trained model bundle
#   produced by SegWise.ipynb and lets users:
#     1. Explore customer segments visually
#     2. Drill into any segment and download its customers
#     3. Predict which segment a brand-new customer belongs to
#     4. Browse raw dataset statistics and correlations
#
# PREREQUISITE:
#   Run ALL cells in notebooks/SegWise.ipynb first.
#   That creates: models/segwise_model.pkl
#                 outputs/cluster_predictions.csv
#                 outputs/cluster_summary.csv
#   Without those files, this app will show an error and stop.
#
# HOW TO RUN:
#   cd D:\AIML\projects\segwise
#   streamlit run app.py
#
# PAGES:
#   Dashboard          — KPI cards, 3D scatter, donut, income bar
#   Cluster Explorer   — drill into any segment + download CSV
#   Predict Customer   — enter new customer → get segment prediction
#   Data Insights      — raw EDA + correlation heatmap + data preview
#
# Author: Karthika Krishna M | SegWise Project
# =============================================================================

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# Each library explained — understand the role before using it
# ─────────────────────────────────────────────────────────────────

import streamlit as st
# streamlit: the entire web app framework
# Every UI element (buttons, sliders, charts, columns) is an st.xxx() call
# Streamlit re-runs this ENTIRE script from top to bottom on every
# user interaction (slider move, button click, page change)
# This is called "reactive execution" — different from normal Python scripts

import pandas as pd
# pandas: DataFrames — used to load CSVs, filter rows, groupby, describe

import numpy as np
# numpy: array operations — used for np.triu (triangle mask), np.vstack,
# np.number (dtype filter), and array shape checks

import matplotlib.pyplot as plt
# matplotlib: base plotting — used for the correlation heatmap on Page 4
# We use matplotlib here because st.pyplot() renders it inline in Streamlit
# For interactive charts (Pages 1-3) we use Plotly instead

import seaborn as sns
# seaborn: statistical visualisation — used for sns.heatmap on Page 4
# Built on top of matplotlib, gives nicer default styling for heatmaps

import pickle
# pickle: deserialise the .pkl bundle from disk back into Python objects
# The bundle contains scaler, pca, ohe, model, persona_map all at once

import plotly.express as px
# plotly.express: high-level interactive chart API — one-line chart creation
# Used for: scatter_3d, pie (donut), bar (horizontal), histogram
# WHY Plotly not Matplotlib for Pages 1-3?
# Plotly charts are interactive: hover tooltips, zoom, pan, 3D rotate
# In a web app, interactivity is the whole point

import plotly.graph_objects as go
# plotly.graph_objects: low-level Plotly API for custom chart construction
# Imported here for flexibility — not used directly in current version
# but available if you want to build custom traces or combine charts

from io import StringIO
# StringIO: treats a string as if it were a file object
# Used when converting a DataFrame to CSV string for the download button
# st.download_button needs bytes/string data, not a DataFrame directly

import warnings
warnings.filterwarnings("ignore")
# suppress: FutureWarning from pandas, ConvergenceWarning from sklearn
# These are non-critical and clutter the terminal output

import datetime
# datetime: dynamically fetch the current year for the copyright footer
# Ensures the app stays current without manual hardcoding


# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
#
# MUST be the VERY FIRST Streamlit command in the file.
# If ANY st.xxx() call runs before this — even st.write() —
# Streamlit raises StreamlitAPIException and the app crashes.
# Rule: set_page_config always goes before everything else.
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SegWise — Customer Intelligence",
    # page_title: text shown in the browser tab
    # Appears as: "SegWise — Customer Intelligence | Streamlit"

    page_icon="SegWise",
    # page_icon: text shown as the favicon in the browser tab

    layout="wide",
    # layout="wide": uses the full browser width (no narrow centre column)
    # layout="centered" (default): content sits in a centred ~700px column
    # For dashboards with multiple charts, "wide" is almost always better

    initial_sidebar_state="expanded"
    # "expanded": sidebar is open when page first loads
    # "collapsed": sidebar is hidden initially (user must click to open)
    # "auto": Streamlit decides based on screen width
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
#
# st.markdown with unsafe_allow_html=True: inject raw HTML + CSS
# WHY inject CSS?
# Streamlit has limited built-in styling. For custom coloured headers,
# card borders, and badge styles, we must write CSS directly.
# unsafe_allow_html=True: tells Streamlit to render the HTML as-is
# instead of escaping it (by default Streamlit escapes HTML for safety)
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Dark gradient header banner ─────────────────────── */
    /* Used in Page 1 Dashboard as the hero banner at the top */
    .segwise-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        /* linear-gradient: smooth colour transition from dark navy → blue */
        /* 135deg: diagonal direction (top-left to bottom-right) */
        padding: 2rem;        /* inner spacing — keeps text away from edges */
        border-radius: 12px;  /* rounded corners */
        color: white;         /* all text inside this div = white */
        margin-bottom: 2rem;  /* spacing below the banner */
        text-align: center;   /* centre all text inside */
    }

    /* ── Left-bordered metric card ────────────────────────── */
    /* Available as a CSS class but not actively used in current version */
    /* You can apply it with: st.markdown('<div class="metric-card">...</div>') */
    .metric-card {
        background: #f8f9fa;           /* light grey background */
        border-left: 4px solid #3498db;/* blue left accent border */
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    /* ── Cluster persona badge ─────────────────────────────── */
    /* Pill-shaped label — apply with unsafe_allow_html=True */
    .cluster-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;  /* high radius = pill/capsule shape */
        font-weight: bold;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# CACHING STRATEGY
#
# WHY caching matters in Streamlit:
# Streamlit re-runs the ENTIRE script on every user interaction.
# Without caching: pickle.load() runs on every slider move → slow.
# With caching: loaded once on first run, reused for all interactions.
#
# TWO cache decorators:
#
# @st.cache_resource
#   → For heavy, non-serialisable objects: ML models, DB connections
#   → Shared across ALL users of the app (one copy in memory)
#   → Persists until the app restarts
#   → Use for: the model bundle (scaler, pca, ohe, model)
#
# @st.cache_data
#   → For DataFrames, arrays, serialisable objects
#   → Hashed by function arguments — returns cached result if args unchanged
#   → Use for: loading CSV files (same path = same result)
# ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(path: str = "models/segwise_model.pkl") -> dict:
    """
    Load the trained model bundle saved by SegWise.ipynb Cell 27.

    Returns a dict with keys:
        scaler      → fitted StandardScaler
        pca         → fitted PCA
        ohe         → fitted OneHotEncoder
        model       → fitted clustering model (KMeans or Agglomerative)
        persona_map → {0: "VIP Shoppers", 1: ...}
        n_clusters  → integer (4)
        best_name   → "K-Means" or "Agglomerative"
        features    → list of column names in training order

    If the .pkl file doesn't exist:
        st.error() → shows a red error box in the UI
        st.stop()  → halts execution cleanly (like sys.exit() for Streamlit)
                     prevents the rest of the app from running with no model
    """
    try:
        with open(path, "rb") as f:
            # "rb" = read binary mode
            # pickle files are binary data, not text — must use "rb" not "r"
            bundle = pickle.load(f)
            # pickle.load: deserialise bytes from file back into Python objects
            # The scaler, pca, ohe, model are fully reconstructed in memory
        return bundle
    except FileNotFoundError:
        st.error(
            "Model not found!\n\n"
            "Run ALL cells in notebooks/SegWise.ipynb first.\n"
            "That creates: models/segwise_model.pkl"
        )
        st.stop()


@st.cache_data
def load_data(path: str = "data/smartcart_customers.csv") -> pd.DataFrame:
    """
    Load the original raw SmartCart dataset.
    Used on Page 4 (Data Insights) for EDA and correlation heatmap.
    Cached so it's only read from disk once per session.
    """
    return pd.read_csv(path)


@st.cache_data
def load_predictions(path: str = "outputs/cluster_predictions.csv") -> pd.DataFrame:
    """
    Load the pre-computed cluster predictions saved by SegWise.ipynb Cell 27.
    Contains every customer row + their Cluster ID + Persona name.
    Used by: Dashboard, Cluster Explorer, and 3D scatter chart.
    """
    return pd.read_csv(path)


# ── Load all three files ──────────────────────────────────────
# These run once and are cached. All subsequent re-runs use cached copies.
bundle  = load_model()
df_raw  = load_data()
df_pred = load_predictions()

# Defensive cleanup for Plotly visualisations
df_pred = df_pred.reset_index(drop=True)
for col in ["Income", "Total_Spending", "Age", "Recency"]:
    if col in df_pred.columns:
        df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce")

# ── Unpack bundle ─────────────────────────────────────────────
# Extract individual objects from the dict for cleaner code below
# These are the EXACT fitted objects from training — not new ones
scaler      = bundle["scaler"]
# StandardScaler fitted on training data
# Has the mean and std of every feature column from training
# We call scaler.transform() (not fit_transform) on new customers

pca         = bundle["pca"]
# PCA fitted on training data
# Has the eigenvectors (directions of max variance) from training
# We call pca.transform() (not fit_transform) on new customers

ohe         = bundle["ohe"]
# OneHotEncoder fitted on training data
# Knows exactly which categories exist for Education and Living_With
# We call ohe.transform() (not fit_transform) on new customers

model       = bundle["model"]
# The winning clustering model (K-Means or Agglomerative)
# Used for predicting which cluster a new customer belongs to

persona_map = bundle["persona_map"]
# Dict: {0: "VIP Shoppers", 1: "Deal Hunters", ...}
# Converts integer cluster IDs to human-readable persona names


# ─────────────────────────────────────────────────────────────────
# SIDEBAR — navigation + model metadata
#
# with st.sidebar: everything inside this block renders in the
# left sidebar panel of the Streamlit app.
# The sidebar stays visible across all pages — good for navigation.
# ─────────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown("## SegWise")
    st.markdown("Customer Intelligence Platform")
    # st.markdown: render text with markdown formatting
    # ## = h2 heading, bold, italic, etc. all supported

    st.markdown("---")
    # "---" in markdown = horizontal divider line
    # Used to visually separate sections in the sidebar

    page = st.radio(
        "Navigate to",
        # label: text above the radio buttons (hidden below with collapsed)
        [
            "Dashboard",
            "Cluster Explorer",
            "Predict New Customer",
            "Data Insights"
        ],
        # options: list of strings — each becomes a selectable radio button
        label_visibility="collapsed"
        # "collapsed": hide the "Navigate to" label above the buttons
        # The buttons are self-explanatory so the label adds no value
        # Other values: "visible" (default), "hidden" (takes space but invisible)
    )
    # page is now one of the 4 strings above
    # We use if/elif blocks below to show the correct page content

    st.markdown("---")

    st.markdown("**Model Info**")
    # **text** in markdown = bold

    st.caption(f"Algorithm : {bundle.get('best_name', 'Agglomerative')}")
    # st.caption: small grey text — good for metadata and secondary info
    # bundle.get('best_name', 'Agglomerative'): safely get value,
    # fall back to 'Agglomerative' if key doesn't exist in older bundles

    st.caption(f"Clusters  : {bundle['n_clusters']}")
    st.caption(f"Customers : {len(df_pred):,}")
    # {:,} format: adds comma as thousands separator → 2,240 not 2240

    st.caption(f"Silhouette: see dashboard")

    # ─────────────────────────────────────────────────────────────────
# PAGE 1: DASHBOARD
# Shows: hero banner → 4 KPI cards → 3D scatter → donut + bar
# ─────────────────────────────────────────────────────────────────

if page == "Dashboard":
    # Streamlit runs this entire elif block when page == "Dashboard"
    # The other elif blocks are skipped completely

    # ── Hero banner ──────────────────────────────────────────────
    # unsafe_allow_html=True: render the <div> as actual HTML
    # The CSS class .segwise-header is defined in the <style> block above
    st.markdown("""
    <div class="segwise-header">
        <h1>SegWise — Customer Intelligence Platform</h1>
        <p>Unsupervised ML-powered customer segmentation for SmartCart</p>
    </div>
    """, unsafe_allow_html=True)

    # ── 4 KPI Cards ──────────────────────────────────────────────
    # st.columns(4): creates 4 equal-width columns side by side
    # We unpack into 4 variables to use "with col:" blocks
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(df_pred):,}"
            # f"{n:,}" → formats number with comma separator
            # 2240 becomes "2,240"
        )

    with col2:
        st.metric(
            label="Segments Found",
            value=bundle["n_clusters"]
            # reads directly from the model bundle
            # shows 4 (or whatever n_clusters was set to in CONFIG)
        )

    with col3:
        # Guard: check column exists before computing mean
        # df_pred might not have Total_Spending if notebook was run differently
        avg_spend = (
            df_pred["Total_Spending"].mean()
            if "Total_Spending" in df_pred.columns
            else 0
        )
        st.metric(
            label="Avg Total Spend",
            value=f"${avg_spend:,.0f}"
            # :,.0f → comma separator, no decimal places, dollar sign prefix
        )

    with col4:
        avg_income = (
            df_pred["Income"].mean()
            if "Income" in df_pred.columns
            else 0
        )
        st.metric(
            label="Avg Annual Income",
            value=f"${avg_income:,.0f}"
        )

    st.markdown("---")

    # ── Interactive 3D Scatter (Plotly) ───────────────────────────
    st.subheader("Customer Segments in 3D Feature Space")
    # st.subheader: renders an h3-level heading in the page

    st.caption("Rotate · Zoom · Hover for details. Each point = one customer.")
    # st.caption: small grey descriptive text below the heading

    if "Persona" in df_pred.columns:
        # Guard: only render if the Persona column was created by the notebook
        # If someone runs app.py before the notebook, df_pred might be empty

        y_col = "Total_Spending" if "Total_Spending" in df_pred.columns else "Recency"
        if y_col == "Recency":
            st.warning("⚠️ Total_Spending column not found — using Recency as fallback. Re-run the notebook.")

        fig_3d = px.scatter_3d(
            df_pred,
            # DataFrame to plot — one row = one point in 3D space

            x="Income",
            # x-axis: Annual household income
            # Higher income customers appear to the right

            y=y_col,
            # y-axis: Total spending across all product categories
            # Fallback to Recency if Total_Spending column missing

            z="Age" if "Age" in df_pred.columns else "NumWebVisitsMonth",
            # z-axis: Customer age
            # Creates a 3D space: Income × Spending × Age

            color="Persona",
            # color: each unique Persona value gets its own colour
            # Plotly automatically creates the colour legend

            title="Customer Segmentation: Income × Spending × Age",

            opacity=0.7,
            # opacity: 0=invisible, 1=solid
            # 0.7 = slightly transparent so overlapping points are visible
            # Without transparency, front points hide all points behind them

            height=560,
            # height in pixels — controls how tall the chart is on screen

            color_discrete_sequence=px.colors.qualitative.Set2
            # Set2: a colour-blind-accessible palette of 8 distinct colours
            # Other options: Plotly, D3, Pastel, Bold, etc.
        )

        fig_3d.update_traces(marker=dict(size=3))
        # update_traces: modify all data traces at once
        # marker size=3: shrink dots from default (6) → less visual clutter
        # with 2000+ points, smaller markers show distribution more clearly

        fig_3d.update_layout(legend=dict(orientation="h", y=-0.12))
        # orientation="h": horizontal legend (left-to-right instead of top-to-bottom)
        # y=-0.12: push legend below the chart so it doesn't overlap the plot

        st.plotly_chart(fig_3d, use_container_width=True)
        # use_container_width=True: chart fills its column width automatically
        # Without this: chart renders at its fixed height × fixed width

    st.markdown("---")

    # ── Donut chart + Avg Income bar side by side ─────────────────
    col_a, col_b = st.columns(2)
    # 2 equal-width columns: donut on left, bar on right

    with col_a:
        st.subheader("Segment Distribution")

        # Count how many customers are in each persona
        seg_counts = (
            df_pred["Persona"]
            .value_counts()          # count per unique Persona value
            .reset_index()           # convert Series → DataFrame
        )
        seg_counts.columns = ["Persona", "Count"]
        # rename columns after reset_index (they default to 0 and "Persona")

        fig_pie = px.pie(
            seg_counts,
            names="Persona",   # column to use for slice labels
            values="Count",    # column to use for slice sizes
            hole=0.4,
            # hole=0.4: creates a donut chart (ring with empty centre)
            # hole=0: full pie chart, hole=1: invisible
            # Donut charts reduce the visual distortion from comparing angles
            # (the human eye compares lengths better than angles)
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_layout(showlegend=True, height=360)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Avg Annual Income by Segment")

        if "Income" in df_pred.columns:
            # Compute average income per persona using groupby
            income_by_seg = (
                df_pred
                .groupby("Persona")["Income"]
                .mean()                    # average income per group
                .reset_index()
                .rename(columns={"Income": "Avg Income"})
                .sort_values("Avg Income") # sort ascending → shortest bar at top
            )

            fig_bar = px.bar(
                income_by_seg,
                x="Avg Income",
                y="Persona",
                orientation="h",
                # orientation="h": horizontal bars
                # Better than vertical when category labels are long strings
                # like "VIP Shoppers" — they don't need to be rotated

                color="Persona",
                # colour each bar differently — matches donut chart colours
                color_discrete_sequence=px.colors.qualitative.Set2,

                height=360
            )
            fig_bar.update_layout(showlegend=False)
            # showlegend=False: the y-axis labels already identify the bars
            # a legend would be redundant here
            st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# PAGE 2: CLUSTER EXPLORER
# Shows: segment selector → stats table → income + spend histograms
#        → download button for filtered CSV
# ─────────────────────────────────────────────────────────────────

elif page == "Cluster Explorer":

    st.title("Cluster Explorer")
    # st.title: renders the largest heading (h1 equivalent)

    st.markdown("Select a segment to explore its statistics and download its customers.")

    if "Persona" in df_pred.columns:

        personas = sorted(df_pred["Persona"].unique())
        # .unique(): get all distinct Persona values as an array
        # sorted(): alphabetical order — consistent dropdown every time

        selected_persona = st.selectbox(
            "Select a customer segment:",
            personas
            # st.selectbox: dropdown menu
            # Returns the currently selected string
            # Streamlit re-runs the script when selection changes
        )

        # ── Filter to selected persona ────────────────────────────
        # Boolean mask: keep only rows where Persona matches selection
        df_segment = df_pred[df_pred["Persona"] == selected_persona].copy()
        # .copy(): create an independent copy so we don't accidentally
        # modify df_pred through the filtered view (pandas SettingWithCopyWarning)

        count = len(df_segment)
        pct   = count / len(df_pred) * 100

        st.markdown(f"### {selected_persona}")
        st.caption(f"{count:,} customers  —  {pct:.1f}% of total base")

        # ── Statistics table ──────────────────────────────────────
        # List only columns that are relevant and actually exist in df_segment
        # Guard against missing columns if notebook produced a different schema
        key_cols = ["Income", "Total_Spending", "Age", "Recency",
                    "NumDealsPurchases", "NumWebVisitsMonth", "Total_Children"]
        numeric_cols = [c for c in key_cols if c in df_segment.columns]
        # List comprehension filter: keeps only columns that exist

        if numeric_cols:
            stats_table = df_segment[numeric_cols].describe().round(2)
            # .describe(): count, mean, std, min, 25%, 50%, 75%, max
            # .round(2): 2 decimal places — cleaner display
            st.dataframe(stats_table, use_container_width=True)
            # st.dataframe: scrollable, sortable interactive table
            # use_container_width=True: fills the available width

        # ── Distribution charts side by side ──────────────────────
        col1, col2 = st.columns(2)

        with col1:
            if "Income" in df_segment.columns:
                fig = px.histogram(
                    df_segment,
                    x="Income",
                    title=f"Income Distribution — {selected_persona}",
                    nbins=30,
                    # nbins=30: number of histogram bars (buckets)
                    # Too few buckets → shape is hidden
                    # Too many buckets → too noisy, individual bars too thin
                    # 30 is a good default for ~200-800 row segments
                    color_discrete_sequence=["#3498db"]
                    # single colour list → all bars the same blue
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "Total_Spending" in df_segment.columns:
                fig = px.histogram(
                    df_segment,
                    x="Total_Spending",
                    title=f"Spending Distribution — {selected_persona}",
                    nbins=30,
                    color_discrete_sequence=["#2ecc71"]
                    # green for spending — visually distinct from income (blue)
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Download button ────────────────────────────────────────
        csv_data = df_segment.to_csv(index=False)
        # .to_csv(index=False): convert DataFrame to CSV-formatted string
        # index=False: don't include the row number column (0, 1, 2...)
        # This string is what the browser downloads as a file

        st.download_button(
            label=f"Download {selected_persona} data as CSV",
            # label: text shown on the button

            data=csv_data,
            # data: the CSV string — Streamlit converts it to downloadable bytes

            file_name=f"segwise_{selected_persona.replace(' ', '_')}.csv",
            # file_name: what the downloaded file is named on the user's computer
            # .replace(' ', '_'): spaces break some file systems → use underscores

            mime="text/csv"
            # mime type: tells the browser this is a CSV file
            # This triggers the correct "Save As" dialog and file association
        )

# ─────────────────────────────────────────────────────────────────
# PAGE 3: PREDICT NEW CUSTOMER
#
# CORE CONCEPT — transform() vs fit_transform():
#
# fit_transform() = learn statistics from THIS data, then apply
# transform()     = use ALREADY LEARNED statistics, just apply
#
# We MUST use transform() here because:
#   scaler was fit on 2200 training customers
#   → mean and std computed from those 2200 customers
#   If we called fit_transform() on 1 new customer:
#   → mean = that customer's value (nonsense)
#   → std  = 0 (only one value, no spread)
#   → every z-score = 0 or NaN → completely wrong prediction
#
# Same logic applies to pca.transform() and ohe.transform()
# ─────────────────────────────────────────────────────────────────

elif page == "Predict New Customer":

    st.title("Predict Customer Segment")
    st.markdown("Fill in the customer's details and click **Predict** to find their segment.")

    # ── Input form ────────────────────────────────────────────────
    # st.form: groups all widgets inside a form boundary
    # WHY use a form here?
    # Without a form: every single slider/input interaction triggers a
    # full script re-run. With 13 input widgets, that's 13 re-runs
    # just to fill out one customer — very slow and laggy.
    # With a form: Streamlit batches ALL inputs and only re-runs
    # the script when the submit button is clicked — one single re-run.

    with st.form("prediction_form"):
        # "prediction_form": unique string key identifying this form
        # Each form in an app needs a different key string

        st.subheader("Customer Demographics")
        col1, col2, col3 = st.columns(3)
        # 3 equal columns: age+income | education+living | children+tenure

        with col1:
            age = st.slider(
                "Age",
                min_value=18,   # youngest realistic customer
                max_value=90,   # matches outlier filter in notebook
                value=40        # default starting value shown on load
            )
            # st.slider: drag-to-select a number in a range
            # Returns an integer

            income = st.number_input(
                "Annual Income ($)",
                min_value=0,
                max_value=600_000,  # matches outlier filter in notebook
                value=50_000,       # realistic default middle income
                step=1_000
                # step=1000: each click of up/down arrow changes by $1000
                # st.number_input: text box with up/down arrows
                # Unlike slider, user can type any value directly
            )

        with col2:
            education = st.selectbox(
                "Education Level",
                ["Graduate", "Postgraduate", "Undergraduate"]
                # Must exactly match the simplified values from notebook Cell 11
                # The OHE was trained on these exact strings
                # If you pass "PhD", OHE has never seen it → all zeros (handle_unknown="ignore")
            )

            living_with = st.selectbox(
                "Living Situation",
                ["Partner", "Alone"]
                # Must exactly match values from notebook Cell 11 living_map
            )

        with col3:
            total_children = st.slider("Total Children", 0, 5, 0)
            # range: 0-5 matches the dataset's actual range (Kidhome + Teenhome)

            tenure_days = st.slider(
                "Days as Customer",
                min_value=0,
                max_value=2000,   # ~5.5 years, matches dataset max tenure
                value=500
            )

        st.subheader("Purchase Behaviour")
        col4, col5, col6 = st.columns(3)

        with col4:
            total_spending = st.number_input(
                "Total Spending ($)",
                min_value=0,
                max_value=3000,  # dataset max Total_Spending
                value=500,
                step=10
            )

            recency = st.slider(
                "Days Since Last Purchase",
                min_value=0,
                max_value=100,   # dataset range for Recency column
                value=30
            )

        with col5:
            num_deals = st.slider("Deal Purchases / month", 0, 15, 3)
            # NumDealsPurchases: how many times they bought using a discount
            # Range 0-15 matches dataset

            num_web = st.slider("Web Purchases / month", 0, 30, 4)
            # NumWebPurchases: completed purchases through the website

        with col6:
            num_catalog = st.slider("Catalog Purchases / month", 0, 30, 2)
            num_store   = st.slider("Store Purchases / month", 0, 20, 5)
            num_web_visits = st.slider("Web Visits / month", 0, 20, 5)
            # NumWebVisitsMonth: visits that did NOT result in a purchase
            # Different from NumWebPurchases — visitors vs buyers

        complain = st.checkbox(
            "Complained in last 2 years",
            value=False
            # Complain column in dataset: 1 = yes, 0 = no
            # Checkbox returns True/False — we convert to 1/0 below
        )

        submitted = st.form_submit_button(
            "Predict Segment",
            use_container_width=True
            # use_container_width=True: button spans full form width
            # Returns True when clicked, False otherwise
        )

    # ── Prediction logic (only runs after button click) ────────
    if submitted:
        # submitted is True only when the form submit button was clicked
        # This entire block is skipped on initial page load

        # Step 1: Build the numeric feature dict
        # Keys must EXACTLY match the column names used during training
        # If a key name is wrong, the column reorder step will fail silently
        new_customer_dict = {
            "Income":              income,
            "Recency":             recency,
            "NumDealsPurchases":   num_deals,
            "NumWebPurchases":     num_web,
            "NumCatalogPurchases": num_catalog,
            "NumStorePurchases":   num_store,
            "NumWebVisitsMonth":   num_web_visits,
            "Complain":            1 if complain else 0,
            # Convert True/False → 1/0 (integer, not boolean)
            # The training data had integer 0 and 1, not Python booleans
            "Response":            0,
            # Response = whether customer accepted last campaign offer
            # We don't know this for a new customer → default to 0 (no)
            "Age":                 age,
            "Customer_Tenure_Days": tenure_days,
            "Total_Spending":      total_spending,
            "Total_Children":      total_children,
        }

        # Step 2: Wrap in a DataFrame
        # ML models expect 2D input: (n_samples, n_features)
        # pd.DataFrame([dict]): the list wrapper creates a single-row DataFrame
        # Without the list, pd.DataFrame(dict) → error (scalar values)
        df_new = pd.DataFrame([new_customer_dict])

        # Step 3: One-hot encode the categorical columns
        # Build a 1-row DataFrame with just the text columns
        cat_input = pd.DataFrame([{
            "Education":  education,
            "Living_With": living_with
        }])

        try:
            encoded_cats = ohe.transform(cat_input)
            # ohe.transform() NOT ohe.fit_transform()
            # transform(): use the EXISTING vocabulary from training
            # fit_transform() here would: learn vocabulary from 1 row → wrong

            cat_df = pd.DataFrame(
                encoded_cats,
                columns=ohe.get_feature_names_out(["Education", "Living_With"])
                # get_feature_names_out: returns ["Education_Graduate",
                # "Education_Postgraduate", "Education_Undergraduate",
                # "Living_With_Alone", "Living_With_Partner"]
            )

            # Step 4: Combine numeric + encoded categorical columns
            # pd.concat axis=1: horizontal join (left = numeric, right = OHE cols)
            df_final = pd.concat([df_new, cat_df], axis=1)

            # Step 5: Align columns to match exact training order
            # The model was trained on columns in a specific order
            # If we pass them in a different order → wrong feature = wrong position
            training_features = bundle["features"]
            # bundle["features"] = the exact column list from df_encoded in notebook

            for col in training_features:
                if col not in df_final.columns:
                    df_final[col] = 0
            # For any column in training that doesn't exist in df_final:
            # add it with value 0 (handles unseen OHE categories gracefully)

            df_final = df_final[training_features]
            # Reorder df_final columns to match training order exactly

            # Step 6: Scale using the SAVED scaler
            X_new_scaled = scaler.transform(df_final)
            # transform(): applies z = (x - training_mean) / training_std
            # Uses the mean and std computed from 2200 training customers
            # NOT the values from this single new customer

            # Step 7: PCA using the SAVED pca
            X_new_pca = pca.transform(X_new_scaled)
            # transform(): projects the new customer onto the EXISTING eigenvectors
            # Same 3D coordinate space as the training customers
            # The clustering model's cluster boundaries are in this same space

            # Step 8: Predict cluster
            if hasattr(model, "predict"):
                # hasattr: check if the model object has a "predict" method
                # KMeans HAS .predict(): computes distance to each centroid → fast
                cluster_id = model.predict(X_new_pca)[0]
                # [0]: take the first (and only) element of the returned array

            else:
                # AgglomerativeClustering does NOT have .predict()
                # WHY? It's a transductive algorithm — it only assigns labels
                # to data it was trained on. It cannot predict for new points.
                #
                # Workaround: refit on all training + new point combined
                # The new point's label is the last element of the result
                # This is O(n²) in memory — slow at large scale but fine for demo
                import numpy as np
                all_pca    = np.vstack([X_new_pca, X_new_pca])
                # vstack: stack arrays vertically → shape (2, 3)
                # We duplicate the new point so fit_predict has ≥2 samples
                all_labels = model.fit_predict(all_pca)
                cluster_id = int(all_labels[0])
                # take index 0 — our new customer

            cluster_id = int(cluster_id)
            # Ensure it's a Python int (not numpy int) for dict lookup
            persona = persona_map.get(cluster_id, f"Cluster {cluster_id}")
            # .get(key, default): return persona name or fallback string

            # ── Display result ─────────────────────────────────────
            st.success(f"### Predicted Segment: {persona}")
            # st.success: green coloured box — used for positive outcomes

            # Marketing strategy lookup (Emojis removed to match request)
            TIPS = {
                "VIP Shoppers":   "Premium loyalty rewards · early product access · concierge offers",
                "Deal Hunters":   "Flash sales · bundle discounts · email coupon campaigns",
                "Dormant Users":  "Win-back discounts · 'We miss you' email sequence · push notifications",
                "Casual Buyers":  "Browse nudges · product discovery emails · loyalty programme onboarding",
            }
            tip = TIPS.get(persona, "Personalised engagement based on spending pattern")

            st.info(
                f"**What this means for SmartCart:**  \n"
                f"This customer belongs to **{persona}**.  \n\n"
                f"**Recommended strategy:** {tip}"
            )
            # st.info: blue coloured box — used for informational content
            # \n\n in markdown = blank line = paragraph break

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            # st.error: red coloured box — used for errors
            st.warning(
                "Common causes:  \n"
                "• SegWise.ipynb was run with a different dataset  \n"
                "• Column names in the bundle don't match current code  \n"
                "• Try: delete models/segwise_model.pkl and re-run the notebook"
            )
            # st.warning: orange coloured box — used for warnings
# ─────────────────────────────────────────────────────────────────
# PAGE 4: DATA INSIGHTS
# Shows: dataset overview → null table → correlation heatmap
#        → raw data preview
# ─────────────────────────────────────────────────────────────────

elif page == "Data Insights":

    st.title("Data Insights")
    st.markdown("Explore the raw SmartCart dataset — shape, nulls, correlations, and a data preview.")

    # ── Dataset overview metrics ──────────────────────────────────
    col1, col2 = st.columns([1, 2])
    # [1, 2]: unequal widths — left column is half the width of right column
    # Useful when one column has a few metrics and another has a table

    with col1:
        st.subheader("Dataset Overview")
        st.metric("Total Rows",    df_raw.shape[0])
        # .shape[0]: number of rows
        st.metric("Total Columns", df_raw.shape[1])
        # .shape[1]: number of columns
        st.metric(
            "Missing Values",
            int(df_raw.isnull().sum().sum())
            # .isnull(): True/False for every cell
            # .sum(): count Trues per column
            # .sum() again: sum all column counts → total missing cells
            # int(): convert from numpy int to Python int for display
        )

    with col2:
        st.subheader("Null Values by Column")
        nulls = (
            df_raw.isnull().sum()
            # .sum(): Series of null counts per column
            .reset_index()
            # reset_index(): turn Series into 2-column DataFrame
        )
        nulls.columns = ["Column", "Null Count"]
        # rename columns (default names after reset_index are 0 and "index")

        nulls = nulls[nulls["Null Count"] > 0]
        # Filter: keep only rows where there are actual nulls
        # Without this filter: all 22 columns shown (most with 0 nulls — useless)

        if len(nulls) > 0:
            st.dataframe(nulls, use_container_width=True, height=150)
            # height=150: fixed pixel height — adds scrollbar if many nulls
        else:
            st.success("Only Income has 24 missing values → filled with median in notebook Cell 9.")

    st.markdown("---")

    # ── Correlation Heatmap ───────────────────────────────────────
    st.subheader("Feature Correlation Matrix")
    st.caption("Red = strong positive correlation · Blue = strong negative · White = no correlation")

    numeric_df = df_raw.select_dtypes(include=np.number)
    # select_dtypes(include=np.number): keep only numeric columns
    # Drops: Education, Marital_Status, Dt_Customer (text columns)
    # .corr() only works on numeric columns — this prevents errors

    fig, ax = plt.subplots(figsize=(12, 8))
    # plt.subplots: create a Figure and Axes object
    # figsize=(12, 8): 12 inches wide × 8 inches tall
    # ax: the Axes object where we draw the heatmap

    corr = numeric_df.corr()
    # .corr(): Pearson correlation matrix
    # Result: n×n DataFrame where entry [i,j] = correlation between col i and col j
    # Range: -1 (perfect negative) to +1 (perfect positive)
    # Diagonal is always 1.0 (a column is perfectly correlated with itself)

    mask = np.triu(np.ones_like(corr, dtype=bool))
    # np.ones_like(corr): array of 1s same shape as corr
    # dtype=bool: convert to True/False
    # np.triu: keep only the upper triangle → True above diagonal, False below
    # sns.heatmap(mask=...): cells where mask=True are hidden (greyed out)
    # WHY mask the upper triangle?
    # The matrix is symmetric: [i,j] == [j,i]
    # Showing both halves is redundant — lower triangle is enough

    sns.heatmap(
        corr,
        mask=mask,
        # hide upper triangle (duplicate info)

        annot=True,
        # annot=True: print the correlation value in each cell
        # Without this you only see the colour — no actual numbers

        annot_kws={"size": 7},
        # size=7: small font so numbers fit inside each cell
        # Default is 10 — too large for a 22×22 matrix

        cmap="coolwarm",
        # colormap: blue (negative) → white (zero) → red (positive)
        # Intuitive: warm colours = things move together, cool = opposite

        fmt=".2f",
        # fmt=".2f": 2 decimal places for annotation numbers
        # Without fmt: might show scientific notation

        ax=ax,
        # ax: which Axes to draw on
        # Always pass ax= explicitly when using plt.subplots()

        linewidths=0.5
        # thin borders between cells — makes grid structure clear
    )

    ax.set_title("SmartCart — Raw Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    # tight_layout(): auto-adjust padding so axis labels don't get clipped

    st.pyplot(fig)
    # st.pyplot(fig): render a matplotlib Figure inline in the Streamlit page
    # WHY st.pyplot not st.plotly_chart?
    # sns.heatmap produces a matplotlib figure — it's not a Plotly chart
    # plt.show() does nothing in Streamlit — you must use st.pyplot()

    plt.close(fig)
    # Close the figure after rendering to free memory
    # Without this: matplotlib accumulates figures in memory across re-runs

    st.markdown("---")

    # ── Raw data table preview ────────────────────────────────────
    st.subheader("Raw Data Preview — First 20 Rows")
    st.dataframe(
        df_raw.head(20),
        # head(20): first 20 rows only — showing all 2240 would be slow
        use_container_width=True
        # st.dataframe: scrollable + sortable interactive table
        # Click column headers to sort ascending/descending
        # Hover over cells to see full values if text is truncated
    )


# ─────────────────────────────────────────────────────────────────
# FOOTER
# Rendered on EVERY page because it's outside all if/elif blocks
# ─────────────────────────────────────────────────────────────────

current_year = datetime.datetime.now().year

st.markdown("---")
st.markdown(
    f"<center style='color: gray; font-size: 0.8em;'>"
    f"&copy; {current_year} SegWise &nbsp;·&nbsp; Built by Karthika Krishna M &nbsp;·&nbsp; "
    f"Unsupervised ML &nbsp;·&nbsp; PCA &nbsp;·&nbsp; "
    f"Agglomerative Clustering &nbsp;·&nbsp; Streamlit"
    f"</center>",
    unsafe_allow_html=True
    # &nbsp; = non-breaking space — gives even spacing between items
    # center tag: horizontally centres the footer text
    # This HTML is simple enough to write inline (no CSS class needed)
)