# =============================================================================
# app.py — SegWise Customer Intelligence (No-Graph Edition)
# Author: Karthika Krishna M
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import datetime

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SegWise — Customer Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# CSS — full custom design, zero graphs needed
# ------------------------------------------------------------------
st.markdown("""
<style>
/* ── Global ───────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #1a1a2e; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* ── Hero banner ──────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.hero h1 { color: #ffffff; font-size: 2rem; margin: 0 0 .4rem; }
.hero p  { color: #a0b4d0; font-size: 1rem; margin: 0; }

/* ── Persona card ─────────────────────────────────────────── */
.persona-card {
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    margin-bottom: .8rem;
    border: 1px solid rgba(255,255,255,0.08);
    transition: transform .15s;
}
.persona-card:hover { transform: translateY(-2px); }
.persona-card h3  { margin: 0 0 .3rem; font-size: 1.05rem; color: #fff; }
.persona-card .kpi-row { display:flex; gap:1rem; flex-wrap:wrap; margin-top:.6rem; }
.persona-card .kpi { background:rgba(0,0,0,.25); border-radius:8px;
                     padding:.35rem .7rem; font-size:.8rem; color:#c8d8e8; }
.persona-card .kpi b { display:block; font-size:1.05rem; color:#fff; }
.persona-card .tip {
    margin-top:.8rem; padding:.5rem .8rem;
    background:rgba(255,255,255,.06); border-radius:8px;
    font-size:.78rem; color:#90a8c0; font-style:italic;
}
.vip    { background: linear-gradient(135deg,#1a3a2a,#0d4a2e); border-color:#2d7a4f; }
.deal   { background: linear-gradient(135deg,#1a2a3a,#0d2a4a); border-color:#2d5a8f; }
.casual { background: linear-gradient(135deg,#2a2a1a,#3a3a0d); border-color:#7a7a2d; }
.dormant{ background: linear-gradient(135deg,#2a1a1a,#3a0d0d); border-color:#7a2d2d; }

/* ── Stat bar (progress-like) ─────────────────────────────── */
.stat-bar-wrap { margin: .25rem 0 .6rem; }
.stat-bar-label { display:flex; justify-content:space-between;
                   font-size:.78rem; color:#a0b0c0; margin-bottom:.2rem; }
.stat-bar-track { background:#1e2a3a; border-radius:99px; height:8px; }
.stat-bar-fill  { height:8px; border-radius:99px; }

/* ── Comparison table ─────────────────────────────────────── */
.cmp-table { width:100%; border-collapse:collapse; font-size:.82rem; color:#c8d8e8; }
.cmp-table th {
    background:#1a2a3a; color:#90b8d8; font-weight:600;
    padding:.55rem .8rem; text-align:left; border-bottom:1px solid #2a3a4a;
}
.cmp-table td { padding:.5rem .8rem; border-bottom:1px solid #1a2232; }
.cmp-table tr:hover td { background:#1a2535; }
.badge {
    display:inline-block; padding:.15rem .55rem; border-radius:99px;
    font-size:.72rem; font-weight:600;
}
.badge-vip    { background:#0d4a2e; color:#4adc8c; }
.badge-deal   { background:#0d2a4a; color:#4a9cdc; }
.badge-casual { background:#3a3a0d; color:#dcdc4a; }
.badge-dormant{ background:#3a0d0d; color:#dc4a4a; }

/* ── Section header ───────────────────────────────────────── */
.sec-header {
    color:#90b8d8; font-size:.72rem; font-weight:700;
    letter-spacing:.12em; text-transform:uppercase;
    margin: 1.2rem 0 .6rem; padding-bottom:.3rem;
    border-bottom:1px solid #1e2a3a;
}

/* ── Predict result ───────────────────────────────────────── */
.result-box {
    border-radius:14px; padding:1.5rem;
    text-align:center; margin-top:1rem;
}
.result-box h2 { margin:0 0 .3rem; color:#fff; font-size:1.6rem; }
.result-box p  { margin:0; color:#a0c8e0; }

/* ── Info row ─────────────────────────────────────────────── */
.info-row {
    display:flex; gap:.6rem; flex-wrap:wrap;
    margin:.8rem 0;
}
.info-chip {
    background:#1a2535; border:1px solid #2a3a4a;
    border-radius:99px; padding:.3rem .8rem;
    font-size:.78rem; color:#90b8d8;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
MODEL_PATH = "models/segwise_model.pkl"
DATA_PATH  = "data/smartcart_customers.csv"
PRED_PATH  = "outputs/cluster_predictions.csv"

PERSONA_NAMES = {
    0: "Casual Buyers",
    1: "VIP Shoppers",
    2: "Deal Hunters",
    3: "Dormant Users",
}

PERSONA_CSS = {
    "VIP Shoppers":  "vip",
    "Deal Hunters":  "deal",
    "Casual Buyers": "casual",
    "Dormant Users": "dormant",
}
PERSONA_BADGE = {
    "VIP Shoppers":  "badge-vip",
    "Deal Hunters":  "badge-deal",
    "Casual Buyers": "badge-casual",
    "Dormant Users": "badge-dormant",
}
PERSONA_EMOJI = {
    "VIP Shoppers":  "",
    "Deal Hunters":  "",
    "Casual Buyers": "",
    "Dormant Users": "",
}
TIPS = {
    "VIP Shoppers":  "Premium loyalty rewards · early product access · concierge offers",
    "Deal Hunters":  "Flash sales · bundle discounts · email coupon campaigns",
    "Dormant Users": "Win-back discounts · re-engagement emails · push notifications",
    "Casual Buyers": "Browse nudges · product discovery emails · loyalty onboarding",
}

# ------------------------------------------------------------------
# AUTO-TRAIN
# ------------------------------------------------------------------
def train_and_save():
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    df = pd.read_csv(DATA_PATH)
    df["Age"] = 2025 - df["Year_Birth"]
    sc = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
    df["Total_Spending"]       = df[[c for c in sc if c in df.columns]].sum(axis=1)
    df["Total_Children"]       = df.get("Kidhome", 0) + df.get("Teenhome", 0)
    df["Dt_Customer"]          = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    df["Customer_Tenure_Days"] = (pd.Timestamp("today") - df["Dt_Customer"]).dt.days
    df["Living_With"] = df["Marital_Status"].map(
        {"Married":"Partner","Together":"Partner","Single":"Alone",
         "Divorced":"Alone","Widow":"Alone","Alone":"Alone"}).fillna("Alone")
    df["Education"] = df["Education"].map(
        {"PhD":"Postgraduate","Master":"Postgraduate","Graduation":"Graduate",
         "Basic":"Undergraduate","2n Cycle":"Undergraduate"}).fillna("Graduate")
    df = df[df["Age"] <= 90].copy()
    df = df[df["Income"] <= 600_000].copy()
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df = df.reset_index(drop=True)

    cat_cols = ["Education","Living_With"]
    num_cols = [c for c in ["Income","Recency","NumDealsPurchases","NumWebPurchases",
                             "NumCatalogPurchases","NumStorePurchases","NumWebVisitsMonth",
                             "Complain","Response","Age","Customer_Tenure_Days",
                             "Total_Spending","Total_Children"] if c in df.columns]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_enc = ohe.fit_transform(df[cat_cols])
    cat_df  = pd.DataFrame(cat_enc, columns=ohe.get_feature_names_out(cat_cols))
    scaler  = StandardScaler()
    num_df  = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)
    df_enc  = pd.concat([num_df, cat_df], axis=1)
    features = df_enc.columns.tolist()
    pca     = PCA(n_components=3, random_state=42)
    X_pca   = pca.fit_transform(df_enc.values)
    km_l    = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_pca)
    ag_l    = AgglomerativeClustering(n_clusters=4).fit_predict(X_pca)
    if silhouette_score(X_pca, km_l) >= silhouette_score(X_pca, ag_l):
        best_model, best_labels, best_name = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_pca), km_l, "K-Means"
    else:
        best_model, best_labels, best_name = AgglomerativeClustering(n_clusters=4), ag_l, "Agglomerative"
    df["Cluster"] = best_labels
    spend_rank   = df.groupby("Cluster")["Total_Spending"].mean().rank(ascending=False).astype(int)
    rank_map     = {1:"VIP Shoppers",2:"Deal Hunters",3:"Casual Buyers",4:"Dormant Users"}
    persona_map  = {int(c): rank_map[int(r)] for c, r in spend_rank.items()}
    df["Persona"] = df["Cluster"].map(persona_map)
    os.makedirs("models", exist_ok=True); os.makedirs("outputs", exist_ok=True)
    df[num_cols + ["Cluster","Persona"]].to_csv(PRED_PATH, index=False)
    bundle = {"scaler":scaler,"pca":pca,"ohe":ohe,"model":best_model,
              "persona_map":persona_map,"n_clusters":4,"best_name":best_name,"features":features}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle

# ------------------------------------------------------------------
# LOAD
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

@st.cache_data
def get_display_df():
    df  = pd.read_csv(PRED_PATH)
    df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce").fillna(0).astype(int)
    bun = get_bundle()
    pm  = bun.get("persona_map") or {}
    pm_generic = any("Segment" in str(v) for v in pm.values()) if pm else True
    eff_map = PERSONA_NAMES if pm_generic else pm
    df["Persona"] = df["Cluster"].map(eff_map).fillna("Segment " + df["Cluster"].astype(str))
    for col in ["Income","Total_Spending","Age","Recency"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data
def get_raw_df():
    return pd.read_csv(DATA_PATH)

bundle     = get_bundle()
df_display = get_display_df()
df_raw     = get_raw_df()
scaler = bundle["scaler"]; pca = bundle["pca"]
ohe = bundle["ohe"];       model = bundle["model"]
_pm = bundle.get("persona_map") or {}
effective_pm = PERSONA_NAMES if (any("Segment" in str(v) for v in _pm.values()) if _pm else True) else _pm

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def fmt_usd(val):
    return f"${val:,.0f}"

def progress_bar(value, max_val, color="#4adc8c", label="", val_label=""):
    pct = min(int(value / max_val * 100), 100) if max_val else 0
    return f"""
    <div class="stat-bar-wrap">
        <div class="stat-bar-label"><span>{label}</span><span>{val_label}</span></div>
        <div class="stat-bar-track">
            <div class="stat-bar-fill" style="width:{pct}%;background:{color};"></div>
        </div>
    </div>"""

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### SegWise")
    st.markdown("Customer Intelligence")
    st.markdown("---")
    page = st.radio("", ["Dashboard","Segments","Predict","Data"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model**")
    st.caption(f"Algorithm : {bundle.get('best_name','—')}")
    st.caption(f"Clusters  : {bundle['n_clusters']}")
    st.caption(f"Customers : {len(df_display):,}")
    st.markdown("---")
    st.markdown("**Segments**")
    for p in sorted(df_display["Persona"].unique()):
        cnt = (df_display["Persona"] == p).sum()
        pct = cnt / len(df_display) * 100
        emoji = PERSONA_EMOJI.get(p, "")
        prefix = f"{emoji} " if emoji else ""
        st.caption(f"{prefix}{p}: {cnt:,} ({pct:.0f}%)")

# ------------------------------------------------------------------
# PAGE 1 — DASHBOARD
# ------------------------------------------------------------------
if page == "Dashboard":

    st.markdown("""
    <div class="hero">
      <h1>SegWise — Customer Intelligence</h1>
      <p>Unsupervised ML-powered segmentation for SmartCart · PCA + K-Means / Agglomerative</p>
    </div>""", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Customers",  f"{len(df_display):,}")
    with c2: st.metric("Segments",         bundle["n_clusters"])
    with c3:
        v = df_display["Income"].mean() if "Income" in df_display.columns else 0
        st.metric("Avg Income", fmt_usd(v))
    with c4:
        v = df_display["Total_Spending"].mean() if "Total_Spending" in df_display.columns else 0
        st.metric("Avg Spending", fmt_usd(v))
    with c5:
        v = df_display["Age"].mean() if "Age" in df_display.columns else 0
        st.metric("Avg Age", f"{v:.0f} yrs")

    st.markdown("---")

    # Segment summary cards
    st.markdown('<p class="sec-header">Segment Overview</p>', unsafe_allow_html=True)

    summary = df_display.groupby("Persona").agg(
        Count=("Cluster","count"),
        Avg_Income=("Income","mean"),
        Avg_Spending=("Total_Spending","mean"),
        Avg_Age=("Age","mean"),
        Avg_Recency=("Recency","mean"),
    ).reset_index()

    max_income   = summary["Avg_Income"].max()
    max_spending = summary["Avg_Spending"].max()

    cols = st.columns(4)
    order = ["VIP Shoppers","Deal Hunters","Casual Buyers","Dormant Users"]
    ordered = [p for p in order if p in summary["Persona"].values]

    for i, persona in enumerate(ordered):
        row = summary[summary["Persona"] == persona].iloc[0]
        css  = PERSONA_CSS.get(persona, "casual")
        emoji = PERSONA_EMOJI.get(persona, "")
        prefix = f"{emoji} " if emoji else ""
        pct  = row["Count"] / len(df_display) * 100

        income_bar   = progress_bar(row["Avg_Income"],   max_income,   "#4adc8c", "Avg Income",   fmt_usd(row["Avg_Income"]))
        spending_bar = progress_bar(row["Avg_Spending"], max_spending, "#4a9cdc", "Avg Spending",  fmt_usd(row["Avg_Spending"]))

        with cols[i]:
            st.markdown(f"""
            <div class="persona-card {css}">
              <h3>{prefix}{persona}</h3>
              <div class="kpi-row">
                <div class="kpi"><b>{row['Count']:,.0f}</b>Customers</div>
                <div class="kpi"><b>{pct:.0f}%</b>Share</div>
                <div class="kpi"><b>{row['Avg_Age']:.0f}</b>Avg Age</div>
              </div>
              {income_bar}
              {spending_bar}
              <div class="tip">{TIPS.get(persona,'')}</div>
            </div>
            """, unsafe_allow_html=True)

    # Comparison table
    st.markdown("---")
    st.markdown('<p class="sec-header">Segment Comparison Table</p>', unsafe_allow_html=True)

    rows_html = ""
    for _, row in summary.sort_values("Avg_Income", ascending=False).iterrows():
        p = row["Persona"]
        badge_cls = PERSONA_BADGE.get(p, "badge-casual")
        emoji = PERSONA_EMOJI.get(p, "")
        prefix = f"{emoji} " if emoji else ""
        recency = row.get("Avg_Recency", 0)
        rows_html += f"""
        <tr>
          <td><span class="badge {badge_cls}">{prefix}{p}</span></td>
          <td>{row['Count']:,.0f}</td>
          <td>{row['Count']/len(df_display)*100:.1f}%</td>
          <td>{fmt_usd(row['Avg_Income'])}</td>
          <td>{fmt_usd(row['Avg_Spending'])}</td>
          <td>{row['Avg_Age']:.0f} yrs</td>
          <td>{recency:.0f} days</td>
        </tr>"""

    st.markdown(f"""
    <table class="cmp-table">
      <thead><tr>
        <th>Segment</th><th>Customers</th><th>Share</th>
        <th>Avg Income</th><th>Avg Spending</th><th>Avg Age</th><th>Avg Recency</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# PAGE 2 — SEGMENTS (drill-down)
# ------------------------------------------------------------------
elif page == "Segments":

    st.title("Segment Explorer")

    personas         = sorted(df_display["Persona"].unique())
    selected_persona = st.selectbox("Select a segment:", personas)
    df_seg = df_display[df_display["Persona"] == selected_persona].copy()

    count = len(df_seg)
    pct   = count / len(df_display) * 100
    emoji = PERSONA_EMOJI.get(selected_persona, "")
    prefix = f"{emoji} " if emoji else ""
    css   = PERSONA_CSS.get(selected_persona, "casual")

    # Hero for selected segment
    st.markdown(f"""
    <div class="persona-card {css}" style="margin-bottom:1.2rem;">
      <h3 style="font-size:1.3rem;">{prefix}{selected_persona}</h3>
      <div class="kpi-row">
        <div class="kpi"><b>{count:,}</b>Customers</div>
        <div class="kpi"><b>{pct:.1f}%</b>of Total</div>
        <div class="kpi"><b>{fmt_usd(df_seg['Income'].mean()) if 'Income' in df_seg.columns else '—'}</b>Avg Income</div>
        <div class="kpi"><b>{fmt_usd(df_seg['Total_Spending'].mean()) if 'Total_Spending' in df_seg.columns else '—'}</b>Avg Spending</div>
        <div class="kpi"><b>{df_seg['Age'].mean():.0f} yrs</b>Avg Age</div>
        <div class="kpi"><b>{df_seg['Recency'].mean():.0f} days</b>Avg Recency</div>
      </div>
      <div class="tip">{TIPS.get(selected_persona,'')}</div>
    </div>""", unsafe_allow_html=True)

    # Stats table
    st.markdown('<p class="sec-header">Descriptive Statistics</p>', unsafe_allow_html=True)
    key_cols     = ["Income","Total_Spending","Age","Recency",
                    "NumDealsPurchases","NumWebVisitsMonth","Total_Children"]
    numeric_cols = [c for c in key_cols if c in df_seg.columns]
    if numeric_cols:
        st.dataframe(
            df_seg[numeric_cols].describe().round(2),
            use_container_width=True
        )

    # Income distribution as text bins
    st.markdown('<p class="sec-header">Income Distribution (Bins)</p>', unsafe_allow_html=True)
    if "Income" in df_seg.columns:
        bins   = [0, 20000, 40000, 60000, 80000, 120000, 200000]
        labels = ["<$20K","$20-40K","$40-60K","$60-80K","$80-120K","$120K+"]
        df_seg["Income_Bin"] = pd.cut(df_seg["Income"], bins=bins, labels=labels, right=False)
        bin_counts = df_seg["Income_Bin"].value_counts().sort_index()
        max_count  = bin_counts.max()

        bar_colors = ["#4adc8c","#4a9cdc","#dcdc4a","#dc4a4a","#c04adc","#dc8c4a"]
        for i, (label, cnt) in enumerate(bin_counts.items()):
            color = bar_colors[i % len(bar_colors)]
            st.markdown(
                progress_bar(cnt, max_count, color, str(label), f"{cnt:,} customers"),
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.download_button(
        label=f"Download {selected_persona} data as CSV",
        data=df_seg.to_csv(index=False),
        file_name=f"segwise_{selected_persona.replace(' ','_')}.csv",
        mime="text/csv"
    )


# ------------------------------------------------------------------
# PAGE 3 — PREDICT
# ------------------------------------------------------------------
elif page == "Predict":

    st.title("Predict Customer Segment")
    st.markdown("Enter a new customer's details to find which segment they belong to.")

    with st.form("predict_form"):
        st.markdown('<p class="sec-header">Demographics</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age    = st.slider("Age", 18, 90, 40)
            income = st.number_input("Annual Income ($)", 0, 600_000, 50_000, step=1_000)
        with c2:
            education   = st.selectbox("Education", ["Graduate","Postgraduate","Undergraduate"])
            living_with = st.selectbox("Living Situation", ["Partner","Alone"])
        with c3:
            total_children = st.slider("Total Children", 0, 5, 0)
            tenure_days    = st.slider("Days as Customer", 0, 2000, 500)

        st.markdown('<p class="sec-header">Purchase Behaviour</p>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            total_spending = st.number_input("Total Spending ($)", 0, 3000, 500, step=10)
            recency        = st.slider("Days Since Last Purchase", 0, 100, 30)
        with c5:
            num_deals = st.slider("Deal Purchases / month", 0, 15, 3)
            num_web   = st.slider("Web Purchases / month",  0, 30, 4)
        with c6:
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
            cid      = int(model.predict(X_pca)[0]) if hasattr(model,"predict") \
                       else int(model.fit_predict(np.vstack([X_pca,X_pca]))[0])
            persona  = effective_pm.get(cid, f"Segment {cid}")
            css      = PERSONA_CSS.get(persona, "casual")
            emoji    = PERSONA_EMOJI.get(persona, "")
            prefix   = f"{emoji} " if emoji else ""
            tip      = TIPS.get(persona, "Personalised engagement")

            # Find similar customers in the same segment
            seg_df    = df_display[df_display["Persona"] == persona]
            seg_count = len(seg_df)
            seg_pct   = seg_count / len(df_display) * 100
            seg_avg_i = seg_df["Income"].mean() if "Income" in seg_df.columns else 0
            seg_avg_s = seg_df["Total_Spending"].mean() if "Total_Spending" in seg_df.columns else 0

            st.markdown(f"""
            <div class="result-box persona-card {css}">
              <h2>{prefix}{persona}</h2>
              <p>This customer belongs to the <strong>{persona}</strong> segment</p>
            </div>""", unsafe_allow_html=True)

            st.markdown('<p class="sec-header">Segment Context</p>', unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns(4)
            with r1: st.metric("Segment Size",    f"{seg_count:,}")
            with r2: st.metric("Share of Base",   f"{seg_pct:.1f}%")
            with r3: st.metric("Segment Avg Income",   fmt_usd(seg_avg_i))
            with r4: st.metric("Segment Avg Spending", fmt_usd(seg_avg_s))

            st.markdown(f"""
            <div class="info-row">
              <span class="info-chip">Strategy</span>
              <span class="info-chip" style="color:#c8d8e8;">{tip}</span>
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ------------------------------------------------------------------
# PAGE 4 — DATA
# ------------------------------------------------------------------
elif page == "Data":

    st.title("Data Insights")

    # Overview metrics
    st.markdown('<p class="sec-header">Dataset Overview</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Rows",     df_raw.shape[0])
    with c2: st.metric("Total Columns",  df_raw.shape[1])
    with c3: st.metric("Missing Values", int(df_raw.isnull().sum().sum()))
    with c4: st.metric("Numeric Cols",   int(df_raw.select_dtypes(include=np.number).shape[1]))

    # Null breakdown
    nulls = df_raw.isnull().sum().reset_index()
    nulls.columns = ["Column","Null Count"]
    nulls = nulls[nulls["Null Count"] > 0]
    if len(nulls):
        st.markdown('<p class="sec-header">Columns with Missing Values</p>', unsafe_allow_html=True)
        st.dataframe(nulls, use_container_width=True, height=120)

    # Correlation as sortable table (no heatmap needed)
    st.markdown("---")
    st.markdown('<p class="sec-header">Top Feature Correlations with Total Spending</p>', unsafe_allow_html=True)

    num_df = df_display.select_dtypes(include=np.number)
    if "Total_Spending" in num_df.columns:
        corr_series = (
            num_df.corr()["Total_Spending"]
            .drop("Total_Spending")
            .abs()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        corr_series.columns = ["Feature","Correlation with Total_Spending"]
        corr_series["Correlation with Total_Spending"] = corr_series["Correlation with Total_Spending"].round(3)
        max_corr = corr_series["Correlation with Total_Spending"].max()

        for _, row in corr_series.iterrows():
            st.markdown(
                progress_bar(row["Correlation with Total_Spending"], max_corr,
                             "#4adc8c", row["Feature"],
                             f'{row["Correlation with Total_Spending"]:.3f}'),
                unsafe_allow_html=True
            )

    # Raw data preview
    st.markdown("---")
    st.markdown('<p class="sec-header">Raw Data Preview</p>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(20), use_container_width=True)


# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
year = datetime.datetime.now().year
st.markdown("---")
st.markdown(
    f"<center style='color:#506070;font-size:.75rem;'>"
    f"© {year} SegWise &nbsp;·&nbsp; Built by Karthika Krishna M "
    f"&nbsp;·&nbsp; Unsupervised ML &nbsp;·&nbsp; PCA &nbsp;·&nbsp; Streamlit"
    f"</center>", unsafe_allow_html=True
)