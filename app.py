# =============================================================================
# app.py — SegWise Streamlit Dashboard (HuggingFace Spaces Edition)
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

st.set_page_config(
    page_title="SegWise — Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .segwise-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        padding: 2rem; border-radius: 12px;
        color: white; margin-bottom: 2rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
MODEL_PATH = "models/segwise_model.pkl"
DATA_PATH  = "data/smartcart_customers.csv"
PRED_PATH  = "outputs/cluster_predictions.csv"

# The authoritative persona map — used whenever the pkl map is empty
# or Persona column contains generic "Segment N" values.
# Assign by total_spending rank: highest spend = VIP Shoppers.
# This mapping is fixed at project level. Change here to rename personas.
PERSONA_NAMES = {
    0: "Casual Buyers",
    1: "VIP Shoppers",
    2: "Deal Hunters",
    3: "Dormant Users",
}

TIPS = {
    "VIP Shoppers":  "Premium loyalty rewards · early product access · concierge offers",
    "Deal Hunters":  "Flash sales · bundle discounts · email coupon campaigns",
    "Dormant Users": "Win-back discounts · re-engagement email sequence · push notifications",
    "Casual Buyers": "Browse nudges · product discovery emails · loyalty onboarding",
}

# ------------------------------------------------------------------
# AUTO-TRAIN: full pipeline if pkl is missing (runs on HuggingFace)
# ------------------------------------------------------------------

def train_and_save():
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    df = pd.read_csv(DATA_PATH)

    # Feature engineering
    df["Age"] = 2025 - df["Year_Birth"]
    spend_cols = ["MntWines","MntFruits","MntMeatProducts",
                  "MntFishProducts","MntSweetProducts","MntGoldProds"]
    df["Total_Spending"] = df[[c for c in spend_cols if c in df.columns]].sum(axis=1)
    df["Total_Children"] = df.get("Kidhome", 0) + df.get("Teenhome", 0)
    df["Dt_Customer"]    = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    df["Customer_Tenure_Days"] = (pd.Timestamp("today") - df["Dt_Customer"]).dt.days

    living_map = {"Married":"Partner","Together":"Partner",
                  "Single":"Alone","Divorced":"Alone","Widow":"Alone","Alone":"Alone"}
    edu_map    = {"PhD":"Postgraduate","Master":"Postgraduate",
                  "Graduation":"Graduate","Basic":"Undergraduate","2n Cycle":"Undergraduate"}
    df["Living_With"] = df["Marital_Status"].map(living_map).fillna("Alone")
    df["Education"]   = df["Education"].map(edu_map).fillna("Graduate")

    df = df[df["Age"] <= 90].copy()
    df = df[df["Income"] <= 600_000].copy()
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df = df.reset_index(drop=True)

    cat_cols = ["Education", "Living_With"]
    num_cols = [c for c in ["Income","Recency","NumDealsPurchases","NumWebPurchases",
                             "NumCatalogPurchases","NumStorePurchases","NumWebVisitsMonth",
                             "Complain","Response","Age","Customer_Tenure_Days",
                             "Total_Spending","Total_Children"] if c in df.columns]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_enc = ohe.fit_transform(df[cat_cols])
    cat_df  = pd.DataFrame(cat_enc, columns=ohe.get_feature_names_out(cat_cols))

    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df[num_cols])
    num_df = pd.DataFrame(num_scaled, columns=num_cols)

    df_enc   = pd.concat([num_df, cat_df], axis=1)
    features = df_enc.columns.tolist()

    pca   = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(df_enc.values)

    n_clusters = 4
    km_labels  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_pca)
    ag_labels  = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_pca)

    if silhouette_score(X_pca, km_labels) >= silhouette_score(X_pca, ag_labels):
        best_model  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_pca)
        best_labels = km_labels
        best_name   = "K-Means"
    else:
        best_model  = AgglomerativeClustering(n_clusters=n_clusters)
        best_labels = ag_labels
        best_name   = "Agglomerative"

    df["Cluster"] = best_labels

    # Assign persona by spending rank (highest spend → VIP Shoppers)
    spend_rank = df.groupby("Cluster")["Total_Spending"].mean().rank(ascending=False).astype(int)
    rank_to_name = {1:"VIP Shoppers", 2:"Deal Hunters", 3:"Casual Buyers", 4:"Dormant Users"}
    persona_map  = {int(cid): rank_to_name[int(rank)] for cid, rank in spend_rank.items()}
    df["Persona"] = df["Cluster"].map(persona_map)

    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Save with REAL values (not scaled)
    df[num_cols + ["Cluster", "Persona"]].to_csv(PRED_PATH, index=False)
    df.groupby("Cluster")[num_cols].mean().round(2).to_csv("outputs/cluster_summary.csv")

    bundle = {"scaler":scaler, "pca":pca, "ohe":ohe, "model":best_model,
              "persona_map":persona_map, "n_clusters":n_clusters,
              "best_name":best_name, "features":features}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


# ------------------------------------------------------------------
# LOAD BUNDLE
# ------------------------------------------------------------------

@st.cache_resource
def get_bundle():
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists(DATA_PATH):
            st.error("data/smartcart_customers.csv not found.")
            st.stop()
        with st.spinner("Training model (~15 s)..."):
            return train_and_save()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------
# LOAD DISPLAY DATAFRAME
# Key insight: cluster_predictions.csv has real values, but Persona
# column is "Segment N" (generic). We always remap using PERSONA_NAMES.
# We convert to plain Python str explicitly to avoid ArrowStringArray
# issues with .str accessor in pandas 2.x.
# ------------------------------------------------------------------

@st.cache_data
def get_display_df():
    df = pd.read_csv(PRED_PATH)

    # Force Cluster to plain int
    df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce").fillna(0).astype(int)

    # Always remap Persona using bundle persona_map first,
    # fall back to PERSONA_NAMES if map is missing/empty.
    bun = get_bundle()
    pm  = bun.get("persona_map") or {}

    # Detect if pm values are generic ("Segment N") or meaningful
    pm_values   = list(pm.values())
    pm_generic  = any("Segment" in str(v) for v in pm_values) if pm_values else True

    effective_map = PERSONA_NAMES if pm_generic else pm

    # ALWAYS remap — don't trust the CSV Persona column at all
    df["Persona"] = df["Cluster"].map(effective_map).fillna(
        "Segment " + df["Cluster"].astype(str)
    )

    # Ensure numeric columns are proper dtypes
    for col in ["Income", "Total_Spending", "Age", "Recency"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data
def get_raw_df():
    return pd.read_csv(DATA_PATH)


# ------------------------------------------------------------------
# BOOTSTRAP
# ------------------------------------------------------------------

bundle     = get_bundle()
df_display = get_display_df()
df_raw     = get_raw_df()

scaler = bundle["scaler"]
pca    = bundle["pca"]
ohe    = bundle["ohe"]
model  = bundle["model"]

# Build effective persona map for prediction page
_raw_pm    = bundle.get("persona_map") or {}
_pm_vals   = list(_raw_pm.values())
_pm_generic = any("Segment" in str(v) for v in _pm_vals) if _pm_vals else True
effective_pm = PERSONA_NAMES if _pm_generic else _raw_pm


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------

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
    st.caption(f"Segments  : {', '.join(df_display['Persona'].unique())}")


# ------------------------------------------------------------------
# PAGE 1: DASHBOARD
# ------------------------------------------------------------------

if page == "Dashboard":

    st.markdown("""
    <div class="segwise-header">
        <h1>SegWise — Customer Intelligence Platform</h1>
        <p>Unsupervised ML-powered customer segmentation for SmartCart</p>
    </div>
    """, unsafe_allow_html=True)

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
    st.subheader("Customer Segments in 3D Feature Space")
    st.caption("Rotate · Zoom · Hover for details. Each point = one customer.")

    y_col   = "Total_Spending" if "Total_Spending" in df_display.columns else "Recency"
    z_col   = "Age"            if "Age"            in df_display.columns else "NumWebVisitsMonth"
    plot_df = df_display[["Income", y_col, z_col, "Persona"]].dropna()

    if len(plot_df) > 0:
        fig_3d = px.scatter_3d(
            plot_df, x="Income", y=y_col, z=z_col, color="Persona",
            title=f"Income × {y_col} × {z_col}",
            opacity=0.7, height=560,
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"Income":"Annual Income ($)",
                    y_col:"Total Spending ($)" if "Spending" in y_col else y_col,
                    z_col:"Age (years)" if z_col=="Age" else z_col,
                    "Persona":"Segment"}
        )
        fig_3d.update_traces(marker=dict(size=3))
        fig_3d.update_layout(legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.warning("No data to plot.")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Segment Distribution")
        seg_counts = df_display["Persona"].value_counts().reset_index()
        seg_counts.columns = ["Persona", "Count"]
        fig_pie = px.pie(
            seg_counts, names="Persona", values="Count", hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_layout(showlegend=True, height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Avg Annual Income by Segment")
        income_seg = (
            df_display.groupby("Persona")["Income"].mean()
            .reset_index().rename(columns={"Income":"Avg Income"})
            .sort_values("Avg Income")
        )
        fig_bar = px.bar(
            income_seg, x="Avg Income", y="Persona", orientation="h",
            color="Persona", color_discrete_sequence=px.colors.qualitative.Set2,
            height=380, labels={"Avg Income":"Avg Annual Income ($)","Persona":"Segment"}
        )
        fig_bar.update_layout(showlegend=False)
        fig_bar.update_xaxes(tickprefix="$", tickformat=",")
        st.plotly_chart(fig_bar, use_container_width=True)


# ------------------------------------------------------------------
# PAGE 2: CLUSTER EXPLORER
# ------------------------------------------------------------------

elif page == "Cluster Explorer":

    st.title("Cluster Explorer")
    st.markdown("Select a segment to explore its statistics and download its customers.")

    personas         = sorted(df_display["Persona"].unique())
    selected_persona = st.selectbox("Select a customer segment:", personas)
    df_seg           = df_display[df_display["Persona"] == selected_persona].copy()

    count = len(df_seg)
    pct   = count / len(df_display) * 100
    st.markdown(f"### {selected_persona}")
    st.caption(f"{count:,} customers — {pct:.1f}% of total base")

    key_cols     = ["Income","Total_Spending","Age","Recency",
                    "NumDealsPurchases","NumWebVisitsMonth","Total_Children"]
    numeric_cols = [c for c in key_cols if c in df_seg.columns]
    if numeric_cols:
        st.dataframe(df_seg[numeric_cols].describe().round(2), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if "Income" in df_seg.columns:
            fig = px.histogram(
                df_seg, x="Income",
                title=f"Income Distribution — {selected_persona}",
                nbins=30, color_discrete_sequence=["#3498db"],
                labels={"Income":"Annual Income ($)"}
            )
            fig.update_xaxes(tickprefix="$", tickformat=",")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "Total_Spending" in df_seg.columns:
            fig = px.histogram(
                df_seg, x="Total_Spending",
                title=f"Spending Distribution — {selected_persona}",
                nbins=30, color_discrete_sequence=["#2ecc71"],
                labels={"Total_Spending":"Total Spending ($)"}
            )
            fig.update_xaxes(tickprefix="$", tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label=f"Download {selected_persona} data as CSV",
        data=df_seg.to_csv(index=False),
        file_name=f"segwise_{selected_persona.replace(' ','_')}.csv",
        mime="text/csv"
    )


# ------------------------------------------------------------------
# PAGE 3: PREDICT NEW CUSTOMER
# ------------------------------------------------------------------

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
            "Income":income, "Recency":recency,
            "NumDealsPurchases":num_deals, "NumWebPurchases":num_web,
            "NumCatalogPurchases":num_catalog, "NumStorePurchases":num_store,
            "NumWebVisitsMonth":num_web_visits,
            "Complain":1 if complain else 0, "Response":0,
            "Age":age, "Customer_Tenure_Days":tenure_days,
            "Total_Spending":total_spending, "Total_Children":total_children,
        }
        df_new    = pd.DataFrame([new_dict])
        cat_input = pd.DataFrame([{"Education":education,"Living_With":living_with}])

        try:
            cat_enc  = ohe.transform(cat_input)
            cat_df   = pd.DataFrame(cat_enc,
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

            persona = effective_pm.get(cid, f"Segment {cid}")
            st.success(f"### Predicted Segment: {persona}")
            st.info(
                f"**Customer belongs to: {persona}**\n\n"
                f"**Recommended strategy:** {TIPS.get(persona, 'Personalised engagement')}"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ------------------------------------------------------------------
# PAGE 4: DATA INSIGHTS
# ------------------------------------------------------------------

elif page == "Data Insights":

    st.title("Data Insights")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Dataset Overview")
        st.metric("Total Rows",     df_raw.shape[0])
        st.metric("Total Columns",  df_raw.shape[1])
        st.metric("Missing Values", int(df_raw.isnull().sum().sum()))
    with col2:
        st.subheader("Null Values by Column")
        nulls = df_raw.isnull().sum().reset_index()
        nulls.columns = ["Column","Null Count"]
        nulls = nulls[nulls["Null Count"] > 0]
        if len(nulls):
            st.dataframe(nulls, use_container_width=True, height=150)
        else:
            st.success("No missing values in the dataset.")

    st.markdown("---")
    st.subheader("Feature Correlation Matrix")
    numeric_df = df_raw.select_dtypes(include=np.number)
    corr       = numeric_df.corr()
    mask       = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax    = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, mask=mask, annot=True, annot_kws={"size":7},
                cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    ax.set_title("SmartCart — Raw Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Raw Data Preview — First 20 Rows")
    st.dataframe(df_raw.head(20), use_container_width=True)


# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------

year = datetime.datetime.now().year
st.markdown("---")
st.markdown(
    f"<center style='color:gray;font-size:0.8em;'>"
    f"&copy; {year} SegWise &nbsp;·&nbsp; Built by Karthika Krishna M "
    f"&nbsp;·&nbsp; Unsupervised ML &nbsp;·&nbsp; PCA &nbsp;·&nbsp; Streamlit"
    f"</center>", unsafe_allow_html=True
)
