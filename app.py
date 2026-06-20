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
# ROOT CAUSE FIX (2025-06-20):
#   cluster_predictions.csv was saved with StandardScaler z-score values,
#   NOT original raw values. Plotly charts rendered blank because
#   z-scored Income values cluster tightly around 0 (range -1 to +1).
#   Fix: load raw values from df_raw and attach Cluster + Persona labels
#   from the predictions CSV. Charts now use real dollar / year values.
#
# Author: Karthika Krishna M | SegWise Project
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings
import datetime

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------
# PAGE CONFIG — must be the very first Streamlit call
# -----------------------------------------------------------------

st.set_page_config(
    page_title="SegWise — Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------------

st.markdown("""
<style>
    .segwise-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .cluster-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------
# CACHING FUNCTIONS
# -----------------------------------------------------------------

@st.cache_resource
def load_model(path: str = "models/segwise_model.pkl") -> dict:
    """
    Load the trained model bundle.
    @cache_resource: loads once, shared across all users/sessions.
    Never use fit_transform() in this app — always transform() only.
    """
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle
    except FileNotFoundError:
        st.error(
            "Model not found!\n\n"
            "Run ALL cells in notebooks/seg-wise.ipynb first.\n"
            "That notebook creates:  models/segwise_model.pkl"
        )
        st.stop()


@st.cache_data
def load_data(path: str = "data/smartcart_customers.csv") -> pd.DataFrame:
    """
    Load the original raw SmartCart dataset.
    Used for the correlation heatmap (Page 4) and as the base
    for building df_display with real dollar/year values.
    """
    return pd.read_csv(path)


@st.cache_data
def load_predictions(path: str = "outputs/cluster_predictions.csv") -> pd.DataFrame:
    """
    Load the cluster label assignments saved by the notebook.

    ROOT CAUSE NOTE:
    This CSV was saved AFTER StandardScaler transform, so Income,
    Total_Spending, Age etc. are z-scores (range roughly -3 to +3),
    NOT their original values. We only use Cluster and Persona columns
    from this file. Real values come from df_raw.
    """
    return pd.read_csv(path)


@st.cache_data
def build_display_df(
    raw_path: str = "data/smartcart_customers.csv",
    pred_path: str = "outputs/cluster_predictions.csv"
) -> pd.DataFrame:
    """
    Build the DataFrame used for ALL Plotly visualisations.

    Strategy:
      1. Load df_raw  — has real Income, real Age, real spending columns
      2. Load df_pred — has Cluster (int) and Persona (str)
      3. Engineer the same derived columns the notebook created
         (Age, Total_Spending, Total_Children, Customer_Tenure_Days)
      4. Apply the same outlier filters so row counts match df_pred
      5. Attach Cluster + Persona from df_pred by positional index

    WHY positional join and not a key join?
    The notebook drops ID after loading and resets the index, so
    df_pred rows are in the same order as df_raw rows (after outlier
    removal). We align on that shared positional order.

    Returns a DataFrame with raw/human-readable values + Cluster + Persona.
    """
    df_raw  = pd.read_csv(raw_path)
    df_pred = pd.read_csv(pred_path)

    # Step 1: Engineer derived columns (same as notebook)
    df_raw["Age"] = 2025 - df_raw["Year_Birth"]

    spend_cols = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    existing_spend = [c for c in spend_cols if c in df_raw.columns]
    if existing_spend:
        df_raw["Total_Spending"] = df_raw[existing_spend].sum(axis=1)

    if "Kidhome" in df_raw.columns and "Teenhome" in df_raw.columns:
        df_raw["Total_Children"] = df_raw["Kidhome"] + df_raw["Teenhome"]

    if "Dt_Customer" in df_raw.columns:
        df_raw["Dt_Customer"] = pd.to_datetime(
            df_raw["Dt_Customer"], dayfirst=True, errors="coerce"
        )
        df_raw["Customer_Tenure_Days"] = (
            pd.Timestamp("today") - df_raw["Dt_Customer"]
        ).dt.days

    # Step 2: Apply same outlier filters as notebook so row count = df_pred
    if "Age" in df_raw.columns:
        df_raw = df_raw[df_raw["Age"] <= 90]
    if "Income" in df_raw.columns:
        df_raw = df_raw[df_raw["Income"] <= 600_000]
        df_raw["Income"] = df_raw["Income"].fillna(df_raw["Income"].median())

    # Step 3: Reset both indexes for a clean positional join
    df_raw  = df_raw.reset_index(drop=True)
    df_pred = df_pred.reset_index(drop=True)

    # Step 4: Trim to shorter length in case of minor mismatch
    n = min(len(df_raw), len(df_pred))
    df_raw  = df_raw.iloc[:n].copy()
    df_pred = df_pred.iloc[:n].copy()

    # Step 5: Attach Cluster and Persona from df_pred
    df_raw["Cluster"] = df_pred["Cluster"].values

    if "Persona" in df_pred.columns:
        df_raw["Persona"] = df_pred["Persona"].values
    else:
        df_raw["Persona"] = "Segment " + df_pred["Cluster"].astype(str)

    return df_raw


# -----------------------------------------------------------------
# LOAD ALL DATA
# -----------------------------------------------------------------

bundle  = load_model()
df_raw  = load_data()         # truly raw CSV — used for heatmap (Page 4)
df_pred = load_predictions()  # z-scored predictions — we only need Cluster + Persona

# Build display DataFrame with REAL (unscaled) values + cluster labels
df_display = build_display_df()

# Overwrite generic "Segment N" Persona names with meaningful names
# from the persona_map stored in the model bundle
persona_map = bundle.get("persona_map", {})
FALLBACK_PERSONA_MAP = {
    0: "Casual Buyers",
    1: "VIP Shoppers",
    2: "Deal Hunters",
    3: "Dormant Users",
}
effective_persona_map = persona_map if persona_map else FALLBACK_PERSONA_MAP

if "Persona" in df_display.columns:
    sample_vals = df_display["Persona"].dropna().unique().tolist()
    is_generic  = any(str(v).startswith("Segment ") for v in sample_vals)
    if is_generic:
        df_display["Persona"] = df_display["Cluster"].map(effective_persona_map)
        df_display["Persona"] = df_display["Persona"].fillna(
            "Segment " + df_display["Cluster"].astype(str)
        )

# Unpack bundle objects
scaler = bundle["scaler"]
pca    = bundle["pca"]
ohe    = bundle["ohe"]
model  = bundle["model"]


# -----------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------

with st.sidebar:
    st.markdown("## SegWise")
    st.markdown("Customer Intelligence Platform")
    st.markdown("---")

    page = st.radio(
        "Navigate to",
        ["Dashboard", "Cluster Explorer", "Predict New Customer", "Data Insights"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Model Info**")
    st.caption(f"Algorithm : {bundle.get('best_name', 'Agglomerative')}")
    st.caption(f"Clusters  : {bundle['n_clusters']}")
    st.caption(f"Customers : {len(df_display):,}")
    st.caption("Silhouette: see dashboard")


# -----------------------------------------------------------------
# PAGE 1: DASHBOARD
# Hero banner → KPI cards → 3D scatter → donut + income bar
#
# KEY FIX: all charts use df_display which has REAL (unscaled) values.
# Before the fix, charts used df_pred which had z-score values
# (Income range -1 to +1), causing Plotly to render blank axes.
# -----------------------------------------------------------------

if page == "Dashboard":

    st.markdown("""
    <div class="segwise-header">
        <h1>SegWise — Customer Intelligence Platform</h1>
        <p>Unsupervised ML-powered customer segmentation for SmartCart</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(df_display):,}")

    with col2:
        st.metric("Segments Found", bundle["n_clusters"])

    with col3:
        avg_spend = (
            df_display["Total_Spending"].mean()
            if "Total_Spending" in df_display.columns else 0
        )
        st.metric("Avg Total Spend", f"${avg_spend:,.0f}")

    with col4:
        avg_income = (
            df_display["Income"].mean()
            if "Income" in df_display.columns else 0
        )
        st.metric("Avg Annual Income", f"${avg_income:,.0f}")

    st.markdown("---")

    # 3D Scatter — Income x Total_Spending x Age
    # df_display has real values: Income in $10k-$90k range, not -1 to +1
    st.subheader("Customer Segments in 3D Feature Space")
    st.caption("Rotate · Zoom · Hover for details. Each point = one customer.")

    if "Persona" in df_display.columns and "Income" in df_display.columns:

        z_col = "Age" if "Age" in df_display.columns else "NumWebVisitsMonth"
        y_col = "Total_Spending" if "Total_Spending" in df_display.columns else "Recency"

        # Drop NaN rows in axes columns — NaN causes Plotly to silently skip traces
        plot_df = df_display[["Income", y_col, z_col, "Persona"]].dropna()

        fig_3d = px.scatter_3d(
            plot_df,
            x="Income",
            y=y_col,
            z=z_col,
            color="Persona",
            title=f"Customer Segmentation: Income x {y_col} x {z_col}",
            opacity=0.7,
            height=580,
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={
                "Income":  "Annual Income ($)",
                y_col:     f"{y_col} ($)" if "Spending" in y_col else y_col,
                z_col:     "Age (years)" if z_col == "Age" else z_col,
                "Persona": "Customer Segment"
            }
        )
        fig_3d.update_traces(marker=dict(size=3))
        fig_3d.update_layout(legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("---")

    # Donut + Income bar side by side
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Segment Distribution")

        seg_counts = (
            df_display["Persona"]
            .value_counts()
            .reset_index()
        )
        seg_counts.columns = ["Persona", "Count"]

        fig_pie = px.pie(
            seg_counts,
            names="Persona",
            values="Count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_layout(showlegend=True, height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Avg Annual Income by Segment")

        if "Income" in df_display.columns:
            income_by_seg = (
                df_display
                .groupby("Persona")["Income"]
                .mean()
                .reset_index()
                .rename(columns={"Income": "Avg Income"})
                .sort_values("Avg Income")
            )

            fig_bar = px.bar(
                income_by_seg,
                x="Avg Income",
                y="Persona",
                orientation="h",
                color="Persona",
                color_discrete_sequence=px.colors.qualitative.Set2,
                height=380,
                labels={"Avg Income": "Avg Annual Income ($)", "Persona": "Segment"}
            )
            fig_bar.update_layout(showlegend=False)
            fig_bar.update_xaxes(tickprefix="$", tickformat=",")
            st.plotly_chart(fig_bar, use_container_width=True)


# -----------------------------------------------------------------
# PAGE 2: CLUSTER EXPLORER
# Segment selector → stats table → histograms → CSV download
# Uses df_display for real unscaled values.
# -----------------------------------------------------------------

elif page == "Cluster Explorer":

    st.title("Cluster Explorer")
    st.markdown("Select a segment to explore its statistics and download its customers.")

    if "Persona" in df_display.columns:

        personas         = sorted(df_display["Persona"].dropna().unique())
        selected_persona = st.selectbox("Select a customer segment:", personas)
        df_segment       = df_display[df_display["Persona"] == selected_persona].copy()

        count = len(df_segment)
        pct   = count / len(df_display) * 100

        st.markdown(f"### {selected_persona}")
        st.caption(f"{count:,} customers  —  {pct:.1f}% of total base")

        # Stats table
        key_cols     = ["Income", "Total_Spending", "Age", "Recency",
                        "NumDealsPurchases", "NumWebVisitsMonth", "Total_Children"]
        numeric_cols = [c for c in key_cols if c in df_segment.columns]

        if numeric_cols:
            st.dataframe(df_segment[numeric_cols].describe().round(2),
                         use_container_width=True)

        # Histograms — now show real $ values, not z-scores
        col1, col2 = st.columns(2)

        with col1:
            if "Income" in df_segment.columns:
                fig = px.histogram(
                    df_segment, x="Income",
                    title=f"Income Distribution — {selected_persona}",
                    nbins=30,
                    color_discrete_sequence=["#3498db"],
                    labels={"Income": "Annual Income ($)"}
                )
                fig.update_xaxes(tickprefix="$", tickformat=",")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "Total_Spending" in df_segment.columns:
                fig = px.histogram(
                    df_segment, x="Total_Spending",
                    title=f"Spending Distribution — {selected_persona}",
                    nbins=30,
                    color_discrete_sequence=["#2ecc71"],
                    labels={"Total_Spending": "Total Spending ($)"}
                )
                fig.update_xaxes(tickprefix="$", tickformat=",")
                st.plotly_chart(fig, use_container_width=True)

        # Download
        csv_data = df_segment.to_csv(index=False)
        st.download_button(
            label=f"Download {selected_persona} data as CSV",
            data=csv_data,
            file_name=f"segwise_{selected_persona.replace(' ', '_')}.csv",
            mime="text/csv"
        )


# -----------------------------------------------------------------
# PAGE 3: PREDICT NEW CUSTOMER
#
# CORE CONCEPT — transform() vs fit_transform():
#   transform()     = use ALREADY LEARNED stats from training data  OK
#   fit_transform() = learn NEW stats from this 1 row = nonsense   WRONG
#
# Pipeline:
#   raw form inputs
#   -> ohe.transform()    (categorical encoding with training vocab)
#   -> scaler.transform() (z-score with training mean/std)
#   -> pca.transform()    (project onto training eigenvectors)
#   -> model.predict()    (distance to training cluster centroids)
#   -> persona_map lookup
# -----------------------------------------------------------------

elif page == "Predict New Customer":

    st.title("Predict Customer Segment")
    st.markdown("Fill in the customer's details and click **Predict** to find their segment.")

    with st.form("prediction_form"):

        st.subheader("Customer Demographics")
        col1, col2, col3 = st.columns(3)

        with col1:
            age    = st.slider("Age", 18, 90, 40)
            income = st.number_input("Annual Income ($)", 0, 600_000, 50_000, step=1_000)

        with col2:
            education   = st.selectbox("Education Level",
                                       ["Graduate", "Postgraduate", "Undergraduate"])
            living_with = st.selectbox("Living Situation", ["Partner", "Alone"])

        with col3:
            total_children = st.slider("Total Children", 0, 5, 0)
            tenure_days    = st.slider("Days as Customer", 0, 2000, 500)

        st.subheader("Purchase Behaviour")
        col4, col5, col6 = st.columns(3)

        with col4:
            total_spending = st.number_input("Total Spending ($)", 0, 3000, 500, step=10)
            recency        = st.slider("Days Since Last Purchase", 0, 100, 30)

        with col5:
            num_deals = st.slider("Deal Purchases / month", 0, 15, 3)
            num_web   = st.slider("Web Purchases / month",  0, 30, 4)

        with col6:
            num_catalog    = st.slider("Catalog Purchases / month", 0, 30, 2)
            num_store      = st.slider("Store Purchases / month",   0, 20, 5)
            num_web_visits = st.slider("Web Visits / month",        0, 20, 5)

        complain  = st.checkbox("Complained in last 2 years", value=False)
        submitted = st.form_submit_button("Predict Segment", use_container_width=True)

    if submitted:

        new_customer_dict = {
            "Income":               income,
            "Recency":              recency,
            "NumDealsPurchases":    num_deals,
            "NumWebPurchases":      num_web,
            "NumCatalogPurchases":  num_catalog,
            "NumStorePurchases":    num_store,
            "NumWebVisitsMonth":    num_web_visits,
            "Complain":             1 if complain else 0,
            "Response":             0,
            "Age":                  age,
            "Customer_Tenure_Days": tenure_days,
            "Total_Spending":       total_spending,
            "Total_Children":       total_children,
        }

        df_new    = pd.DataFrame([new_customer_dict])
        cat_input = pd.DataFrame([{"Education": education, "Living_With": living_with}])

        try:
            encoded_cats = ohe.transform(cat_input)
            cat_df = pd.DataFrame(
                encoded_cats,
                columns=ohe.get_feature_names_out(["Education", "Living_With"])
            )

            df_final = pd.concat([df_new, cat_df], axis=1)

            training_features = bundle["features"]
            for col in training_features:
                if col not in df_final.columns:
                    df_final[col] = 0
            df_final = df_final[training_features]

            X_new_scaled = scaler.transform(df_final)
            X_new_pca    = pca.transform(X_new_scaled)

            if hasattr(model, "predict"):
                cluster_id = model.predict(X_new_pca)[0]
            else:
                all_pca    = np.vstack([X_new_pca, X_new_pca])
                all_labels = model.fit_predict(all_pca)
                cluster_id = int(all_labels[0])

            cluster_id = int(cluster_id)
            persona    = effective_persona_map.get(cluster_id, f"Segment {cluster_id}")

            st.success(f"### Predicted Segment: {persona}")

            TIPS = {
                "VIP Shoppers":  "Premium loyalty rewards · early product access · concierge offers",
                "Deal Hunters":  "Flash sales · bundle discounts · email coupon campaigns",
                "Dormant Users": "Win-back discounts · We miss you email sequence · push notifications",
                "Casual Buyers": "Browse nudges · product discovery emails · loyalty programme onboarding",
            }
            tip = TIPS.get(persona, "Personalised engagement based on spending pattern")

            st.info(
                f"**What this means for SmartCart:**  \n"
                f"This customer belongs to **{persona}**.  \n\n"
                f"**Recommended strategy:** {tip}"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.warning(
                "Common causes:  \n"
                "* SegWise.ipynb was run with a different dataset  \n"
                "* Column names in the bundle do not match current code  \n"
                "* Try: delete models/segwise_model.pkl and re-run the notebook"
            )


# -----------------------------------------------------------------
# PAGE 4: DATA INSIGHTS
# Dataset overview -> nulls -> Pearson heatmap -> raw data preview
# Uses df_raw (truly raw, no feature engineering) for the heatmap.
# -----------------------------------------------------------------

elif page == "Data Insights":

    st.title("Data Insights")
    st.markdown("Explore the raw SmartCart dataset — shape, nulls, correlations, and a data preview.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Dataset Overview")
        st.metric("Total Rows",     df_raw.shape[0])
        st.metric("Total Columns",  df_raw.shape[1])
        st.metric("Missing Values", int(df_raw.isnull().sum().sum()))

    with col2:
        st.subheader("Null Values by Column")
        nulls = df_raw.isnull().sum().reset_index()
        nulls.columns = ["Column", "Null Count"]
        nulls = nulls[nulls["Null Count"] > 0]

        if len(nulls) > 0:
            st.dataframe(nulls, use_container_width=True, height=150)
        else:
            st.success("Only Income has 24 missing values — filled with median in the notebook.")

    st.markdown("---")

    st.subheader("Feature Correlation Matrix")
    st.caption("Red = strong positive correlation · Blue = strong negative · White = no correlation")

    numeric_df = df_raw.select_dtypes(include=np.number)
    corr       = numeric_df.corr()
    mask       = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr, mask=mask,
        annot=True, annot_kws={"size": 7},
        cmap="coolwarm", fmt=".2f",
        ax=ax, linewidths=0.5
    )
    ax.set_title("SmartCart — Raw Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.subheader("Raw Data Preview — First 20 Rows")
    st.dataframe(df_raw.head(20), use_container_width=True)


# -----------------------------------------------------------------
# FOOTER — rendered on every page
# -----------------------------------------------------------------

current_year = datetime.datetime.now().year
st.markdown("---")
st.markdown(
    f"<center style='color: gray; font-size: 0.8em;'>"
    f"&copy; {current_year} SegWise &nbsp;·&nbsp; Built by Karthika Krishna M &nbsp;·&nbsp; "
    f"Unsupervised ML &nbsp;·&nbsp; PCA &nbsp;·&nbsp; "
    f"Agglomerative Clustering &nbsp;·&nbsp; Streamlit"
    f"</center>",
    unsafe_allow_html=True
)
