# =============================================================================
# app.py — SegWise Streamlit Dashboard (HuggingFace Spaces Edition)
#
# SELF-CONTAINED: If models/segwise_model.pkl is missing, the app trains
# the model automatically from data/smartcart_customers.csv before loading.
# No separate notebook run required on HuggingFace.
#
# Author: Karthika Krishna M | SegWise Project
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import plotly.express as px
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
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------
# AUTO-TRAIN: build model bundle if pkl is missing
# This runs on HuggingFace where the notebook cannot be executed.
# -----------------------------------------------------------------

def train_and_save_bundle(
    data_path:  str = "data/smartcart_customers.csv",
    model_path: str = "models/segwise_model.pkl",
    pred_path:  str = "outputs/cluster_predictions.csv",
    summ_path:  str = "outputs/cluster_summary.csv",
):
    """
    Full training pipeline — mirrors notebooks/seg-wise.ipynb exactly.
    Called automatically when segwise_model.pkl does not exist.
    """
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    # ── Load ──────────────────────────────────────────────────────
    df = pd.read_csv(data_path)

    # ── Feature engineering ───────────────────────────────────────
    df["Age"] = 2025 - df["Year_Birth"]
    spend_cols = ["MntWines","MntFruits","MntMeatProducts",
                  "MntFishProducts","MntSweetProducts","MntGoldProds"]
    df["Total_Spending"]       = df[[c for c in spend_cols if c in df.columns]].sum(axis=1)
    df["Total_Children"]       = df.get("Kidhome", 0) + df.get("Teenhome", 0)
    df["Dt_Customer"]          = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    df["Customer_Tenure_Days"] = (pd.Timestamp("today") - df["Dt_Customer"]).dt.days

    living_map = {
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone", "Widow": "Alone", "Alone": "Alone"
    }
    edu_map = {
        "PhD": "Postgraduate", "Master": "Postgraduate",
        "Graduation": "Graduate",
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate"
    }
    df["Living_With"] = df["Marital_Status"].map(living_map).fillna("Alone")
    df["Education"]   = df["Education"].map(edu_map).fillna("Graduate")

    # ── Clean ─────────────────────────────────────────────────────
    df = df[df["Age"] <= 90]
    df = df[df["Income"] <= 600_000]
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df = df.reset_index(drop=True)

    # ── Preprocessing ─────────────────────────────────────────────
    cat_cols  = ["Education", "Living_With"]
    num_cols  = ["Income","Recency","NumDealsPurchases","NumWebPurchases",
                 "NumCatalogPurchases","NumStorePurchases","NumWebVisitsMonth",
                 "Complain","Response","Age","Customer_Tenure_Days",
                 "Total_Spending","Total_Children"]
    num_cols  = [c for c in num_cols if c in df.columns]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_encoded = ohe.fit_transform(df[cat_cols])
    cat_df      = pd.DataFrame(
        cat_encoded,
        columns=ohe.get_feature_names_out(cat_cols)
    )

    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df[num_cols])
    num_df     = pd.DataFrame(num_scaled, columns=num_cols)

    df_encoded = pd.concat([num_df, cat_df], axis=1)
    features   = df_encoded.columns.tolist()

    # ── PCA ───────────────────────────────────────────────────────
    pca    = PCA(n_components=3, random_state=42)
    X_pca  = pca.fit_transform(df_encoded.values)

    # ── Best K via silhouette ──────────────────────────────────────
    # Fixed at K=4 (validated by elbow + silhouette in the notebook)
    n_clusters = 4

    # Compare KMeans vs Agglomerative
    km  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_pca)
    km_sil    = silhouette_score(X_pca, km_labels)

    ag  = AgglomerativeClustering(n_clusters=n_clusters)
    ag_labels = ag.fit_predict(X_pca)
    ag_sil    = silhouette_score(X_pca, ag_labels)

    if km_sil >= ag_sil:
        best_model  = km
        best_labels = km_labels
        best_name   = "K-Means"
    else:
        best_model  = ag
        best_labels = ag_labels
        best_name   = "Agglomerative"

    # ── Persona mapping ────────────────────────────────────────────
    # Determine persona identity from cluster mean stats
    df["Cluster"] = best_labels
    summary       = df.groupby("Cluster")[["Income","Total_Spending"]].mean()

    # Sort clusters by Income to assign personas consistently
    sorted_by_income  = summary["Income"].sort_values()
    sorted_by_spending = summary["Total_Spending"].sort_values()

    cluster_ids = summary.index.tolist()
    # High income + high spend -> VIP
    # High income + high spend (deal) -> Deal Hunters
    # Low income + low spend + active web -> Casual
    # Low income + low spend + less active -> Dormant
    # Simple heuristic: rank by total_spending descending
    spending_rank = summary["Total_Spending"].rank(ascending=False)

    persona_map = {}
    for cid in cluster_ids:
        rank = int(spending_rank[cid])
        if rank == 1:
            persona_map[cid] = "VIP Shoppers"
        elif rank == 2:
            persona_map[cid] = "Deal Hunters"
        elif rank == 3:
            persona_map[cid] = "Casual Buyers"
        else:
            persona_map[cid] = "Dormant Users"

    df["Persona"] = df["Cluster"].map(persona_map)

    # ── Save outputs ───────────────────────────────────────────────
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Save cluster_predictions.csv with REAL (unscaled) values
    # so the app can plot correct dollar amounts directly
    save_cols = num_cols + ["Cluster", "Persona"]
    df[save_cols].to_csv(pred_path, index=False)

    summary_out = df.groupby("Cluster")[num_cols].mean().round(2)
    summary_out["Persona"] = summary_out.index.map(persona_map)
    summary_out.to_csv(summ_path)

    bundle = {
        "scaler":      scaler,
        "pca":         pca,
        "ohe":         ohe,
        "model":       best_model,
        "persona_map": persona_map,
        "n_clusters":  n_clusters,
        "best_name":   best_name,
        "features":    features,
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    return bundle


# -----------------------------------------------------------------
# LOAD OR TRAIN
# -----------------------------------------------------------------

MODEL_PATH = "models/segwise_model.pkl"
DATA_PATH  = "data/smartcart_customers.csv"
PRED_PATH  = "outputs/cluster_predictions.csv"


@st.cache_resource
def get_bundle():
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists(DATA_PATH):
            st.error("data/smartcart_customers.csv not found. Please upload the dataset.")
            st.stop()
        with st.spinner("Training model for the first time... (~15 seconds)"):
            return train_and_save_bundle()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def get_display_df():
    """
    Load cluster_predictions.csv.
    This CSV is now always saved with REAL (unscaled) values by
    train_and_save_bundle(), so Income shows actual dollar amounts.

    For legacy CSVs that may have z-scores: we detect and fix automatically.
    """
    df = pd.read_csv(PRED_PATH)

    # Auto-detect if Income looks like z-scores (all values between -5 and 5)
    # If so, reload from raw CSV and reattach labels
    if "Income" in df.columns:
        income_max = df["Income"].abs().max()
        if income_max < 10:
            # Income is z-scored — rebuild from raw CSV
            df = _rebuild_from_raw(df)

    # Remap generic "Segment N" persona names
    if "Persona" in df.columns:
        is_generic = df["Persona"].str.startswith("Segment ").any()
        if is_generic and "Cluster" in df.columns:
            bun = get_bundle()
            pm  = bun.get("persona_map", {0:"Casual Buyers",1:"VIP Shoppers",
                                           2:"Deal Hunters",3:"Dormant Users"})
            df["Persona"] = df["Cluster"].map(pm).fillna(df["Persona"])

    return df


def _rebuild_from_raw(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Called only when cluster_predictions.csv has z-scored values (legacy).
    Loads real values from smartcart_customers.csv and reattaches labels.
    """
    df_raw = pd.read_csv(DATA_PATH)
    df_raw["Age"] = 2025 - df_raw["Year_Birth"]
    spend_cols = ["MntWines","MntFruits","MntMeatProducts",
                  "MntFishProducts","MntSweetProducts","MntGoldProds"]
    df_raw["Total_Spending"] = df_raw[[c for c in spend_cols if c in df_raw.columns]].sum(axis=1)
    df_raw["Total_Children"] = df_raw.get("Kidhome", 0) + df_raw.get("Teenhome", 0)
    if "Dt_Customer" in df_raw.columns:
        df_raw["Dt_Customer"] = pd.to_datetime(df_raw["Dt_Customer"], dayfirst=True, errors="coerce")
        df_raw["Customer_Tenure_Days"] = (pd.Timestamp("today") - df_raw["Dt_Customer"]).dt.days
    df_raw = df_raw[df_raw["Age"] <= 90]
    df_raw = df_raw[df_raw["Income"] <= 600_000]
    df_raw["Income"] = df_raw["Income"].fillna(df_raw["Income"].median())
    df_raw  = df_raw.reset_index(drop=True)
    df_pred = df_pred.reset_index(drop=True)
    n = min(len(df_raw), len(df_pred))
    df_raw  = df_raw.iloc[:n].copy()
    df_pred = df_pred.iloc[:n].copy()
    df_raw["Cluster"] = df_pred["Cluster"].values
    df_raw["Persona"] = df_pred["Persona"].values if "Persona" in df_pred.columns \
                        else "Segment " + df_pred["Cluster"].astype(str)
    return df_raw


@st.cache_data
def get_raw_df():
    return pd.read_csv(DATA_PATH)


# -----------------------------------------------------------------
# BOOTSTRAP: ensure model + predictions exist before rendering UI
# -----------------------------------------------------------------

bundle     = get_bundle()
df_display = get_display_df()
df_raw     = get_raw_df()

# Unpack bundle
scaler      = bundle["scaler"]
pca         = bundle["pca"]
ohe         = bundle["ohe"]
model       = bundle["model"]
persona_map = bundle.get("persona_map", {0:"Casual Buyers",1:"VIP Shoppers",
                                         2:"Deal Hunters",3:"Dormant Users"})


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


# -----------------------------------------------------------------
# PAGE 1: DASHBOARD
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
        val = df_display["Total_Spending"].mean() if "Total_Spending" in df_display.columns else 0
        st.metric("Avg Total Spend", f"${val:,.0f}")
    with col4:
        val = df_display["Income"].mean() if "Income" in df_display.columns else 0
        st.metric("Avg Annual Income", f"${val:,.0f}")

    st.markdown("---")

    # 3D Scatter
    st.subheader("Customer Segments in 3D Feature Space")
    st.caption("Rotate · Zoom · Hover for details. Each point = one customer.")

    y_col = "Total_Spending" if "Total_Spending" in df_display.columns else "Recency"
    z_col = "Age" if "Age" in df_display.columns else "NumWebVisitsMonth"

    plot_df = df_display[["Income", y_col, z_col, "Persona"]].dropna()

    if len(plot_df) > 0:
        fig_3d = px.scatter_3d(
            plot_df,
            x="Income", y=y_col, z=z_col,
            color="Persona",
            title=f"Customer Segmentation: Income × {y_col} × {z_col}",
            opacity=0.7,
            height=580,
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={
                "Income":  "Annual Income ($)",
                y_col:     "Total Spending ($)" if "Spending" in y_col else y_col,
                z_col:     "Age (years)"         if z_col == "Age"      else z_col,
                "Persona": "Customer Segment"
            }
        )
        fig_3d.update_traces(marker=dict(size=3))
        fig_3d.update_layout(legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.warning("No data available for 3D chart.")

    st.markdown("---")

    # Donut + Income Bar
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Segment Distribution")
        seg_counts = df_display["Persona"].value_counts().reset_index()
        seg_counts.columns = ["Persona", "Count"]
        fig_pie = px.pie(
            seg_counts, names="Persona", values="Count",
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_layout(showlegend=True, height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Avg Annual Income by Segment")
        if "Income" in df_display.columns:
            income_seg = (
                df_display.groupby("Persona")["Income"]
                .mean().reset_index()
                .rename(columns={"Income": "Avg Income"})
                .sort_values("Avg Income")
            )
            fig_bar = px.bar(
                income_seg, x="Avg Income", y="Persona",
                orientation="h", color="Persona",
                color_discrete_sequence=px.colors.qualitative.Set2,
                height=380,
                labels={"Avg Income": "Avg Annual Income ($)", "Persona": "Segment"}
            )
            fig_bar.update_layout(showlegend=False)
            fig_bar.update_xaxes(tickprefix="$", tickformat=",")
            st.plotly_chart(fig_bar, use_container_width=True)


# -----------------------------------------------------------------
# PAGE 2: CLUSTER EXPLORER
# -----------------------------------------------------------------

elif page == "Cluster Explorer":

    st.title("Cluster Explorer")
    st.markdown("Select a segment to explore its statistics and download its customers.")

    personas         = sorted(df_display["Persona"].dropna().unique())
    selected_persona = st.selectbox("Select a customer segment:", personas)
    df_seg           = df_display[df_display["Persona"] == selected_persona].copy()

    count = len(df_seg)
    pct   = count / len(df_display) * 100
    st.markdown(f"### {selected_persona}")
    st.caption(f"{count:,} customers — {pct:.1f}% of total base")

    # Stats table
    key_cols     = ["Income","Total_Spending","Age","Recency",
                    "NumDealsPurchases","NumWebVisitsMonth","Total_Children"]
    numeric_cols = [c for c in key_cols if c in df_seg.columns]
    if numeric_cols:
        st.dataframe(df_seg[numeric_cols].describe().round(2), use_container_width=True)

    # Histograms
    col1, col2 = st.columns(2)
    with col1:
        if "Income" in df_seg.columns:
            fig = px.histogram(
                df_seg, x="Income",
                title=f"Income Distribution — {selected_persona}",
                nbins=30, color_discrete_sequence=["#3498db"],
                labels={"Income": "Annual Income ($)"}
            )
            fig.update_xaxes(tickprefix="$", tickformat=",")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "Total_Spending" in df_seg.columns:
            fig = px.histogram(
                df_seg, x="Total_Spending",
                title=f"Spending Distribution — {selected_persona}",
                nbins=30, color_discrete_sequence=["#2ecc71"],
                labels={"Total_Spending": "Total Spending ($)"}
            )
            fig.update_xaxes(tickprefix="$", tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

    # Download
    st.download_button(
        label=f"Download {selected_persona} data as CSV",
        data=df_seg.to_csv(index=False),
        file_name=f"segwise_{selected_persona.replace(' ','_')}.csv",
        mime="text/csv"
    )


# -----------------------------------------------------------------
# PAGE 3: PREDICT NEW CUSTOMER
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
                                       ["Graduate","Postgraduate","Undergraduate"])
            living_with = st.selectbox("Living Situation", ["Partner","Alone"])
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
        new_dict = {
            "Income": income, "Recency": recency,
            "NumDealsPurchases": num_deals, "NumWebPurchases": num_web,
            "NumCatalogPurchases": num_catalog, "NumStorePurchases": num_store,
            "NumWebVisitsMonth": num_web_visits,
            "Complain": 1 if complain else 0, "Response": 0,
            "Age": age, "Customer_Tenure_Days": tenure_days,
            "Total_Spending": total_spending, "Total_Children": total_children,
        }
        df_new    = pd.DataFrame([new_dict])
        cat_input = pd.DataFrame([{"Education": education, "Living_With": living_with}])

        try:
            cat_encoded = ohe.transform(cat_input)
            cat_df      = pd.DataFrame(cat_encoded,
                            columns=ohe.get_feature_names_out(["Education","Living_With"]))
            df_final = pd.concat([df_new, cat_df], axis=1)

            for col in bundle["features"]:
                if col not in df_final.columns:
                    df_final[col] = 0
            df_final = df_final[bundle["features"]]

            X_scaled = scaler.transform(df_final)
            X_pca    = pca.transform(X_scaled)

            if hasattr(model, "predict"):
                cid = int(model.predict(X_pca)[0])
            else:
                cid = int(model.fit_predict(np.vstack([X_pca, X_pca]))[0])

            persona = persona_map.get(cid, f"Segment {cid}")
            st.success(f"### Predicted Segment: {persona}")

            TIPS = {
                "VIP Shoppers":  "Premium loyalty rewards · early product access · concierge offers",
                "Deal Hunters":  "Flash sales · bundle discounts · email coupon campaigns",
                "Dormant Users": "Win-back discounts · re-engagement email sequence · push notifications",
                "Casual Buyers": "Browse nudges · product discovery emails · loyalty onboarding",
            }
            st.info(
                f"**Customer belongs to: {persona}**\n\n"
                f"**Recommended strategy:** {TIPS.get(persona, 'Personalised engagement')}"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.warning("Column names in the bundle may not match. Re-run training by deleting models/segwise_model.pkl.")


# -----------------------------------------------------------------
# PAGE 4: DATA INSIGHTS
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
        if len(nulls):
            st.dataframe(nulls, use_container_width=True, height=150)
        else:
            st.success("No missing values in the dataset.")

    st.markdown("---")
    st.subheader("Feature Correlation Matrix")
    st.caption("Red = strong positive · Blue = strong negative · White = no correlation")

    numeric_df = df_raw.select_dtypes(include=np.number)
    corr       = numeric_df.corr()
    mask       = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, mask=mask, annot=True, annot_kws={"size": 7},
                cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    ax.set_title("SmartCart — Raw Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Raw Data Preview — First 20 Rows")
    st.dataframe(df_raw.head(20), use_container_width=True)


# -----------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------

current_year = datetime.datetime.now().year
st.markdown("---")
st.markdown(
    f"<center style='color:gray;font-size:0.8em;'>"
    f"&copy; {current_year} SegWise &nbsp;·&nbsp; Built by Karthika Krishna M "
    f"&nbsp;·&nbsp; Unsupervised ML &nbsp;·&nbsp; PCA "
    f"&nbsp;·&nbsp; Agglomerative Clustering &nbsp;·&nbsp; Streamlit"
    f"</center>",
    unsafe_allow_html=True
)
