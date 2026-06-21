
<div align="center">

# SegWise — Customer Intelligence Platform

### Unsupervised ML-powered E-commerce Customer Segmentation

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Discover who your customers really are.**
> SegWise applies PCA + K-Means / Agglomerative Clustering to transform 2,240 raw SmartCart customer records into four actionable business personas — then surfaces them in an interactive Streamlit dashboard with live segment prediction for new customers.

</div>

---

## Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Key Features](#-key-features)
- [Overall Architecture](#-overall-architecture)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack--complete-breakdown)
- [ML Pipeline](#-ml-pipeline)
- [Request Lifecycle](#-request-lifecycle)
- [Data Flow](#-data-flow)
- [UML Diagrams](#-uml-diagrams)
- [DFD Diagrams](#-data-flow-diagrams)
- [Folder Structure](#-folder-structure)
- [Dataset](#-dataset)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [Page Reference](#-page-reference)
- [Data Schema](#-data-schema)
- [Engineering Decisions](#-engineering-decisions--tradeoffs)
- [Security Considerations](#-security-considerations)
- [Performance Optimizations](#-performance-optimizations)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🎯 Problem Statement

E-commerce businesses like SmartCart accumulate vast customer behavioural data — purchases, income, spend categories, visit frequency — but struggle to move beyond one-size-fits-all marketing. Sending the same promotional email to a high-income power buyer and a dormant budget shopper wastes spend and damages retention.

**The gap:** No structure exists in the raw customer database to distinguish high-value segments from disengaged ones. Business teams cannot personalise at scale without a data-driven segmentation layer.

---

## 💡 Solution Overview

SegWise is a two-phase system:

1. **Offline Training Pipeline** (`notebooks/seg-wise.ipynb`): A deterministic, reproducible Jupyter notebook that ingests the SmartCart CSV, engineers features, reduces dimensionality with PCA, evaluates K-Means vs Agglomerative Clustering using elbow + silhouette scoring, and serialises the winning model bundle to `models/segwise_model.pkl`.

2. **Interactive Dashboard** (`app.py`): A Streamlit web app that loads the saved bundle, visualises all four customer segments in 3D, lets analysts drill into any persona, and predicts which segment a brand-new customer belongs to — in real time.

**Output personas discovered:**

| Cluster | Persona | Key Trait |
|---------|---------|-----------|
| 0 | Casual Buyers | Low income · Low spend · Web browsers |
| 1 | VIP Shoppers | High income · High spend · Catalog buyers |
| 2 | Deal Hunters | Mid income · High spend · Discount-driven |
| 3 | Dormant Users | Low income · Very low spend · Infrequent |

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **3D Scatter Visualisation** | Rotate and explore all customers in Income × Spending × Age space with Plotly |
| **Elbow + Silhouette Validation** | Optimal K discovered automatically using KneeLocator — no manual guessing |
| **Dual Algorithm Comparison** | Both K-Means and Agglomerative Clustering trained; best by silhouette score wins |
| **PCA Dimensionality Reduction** | 17-feature space compressed to 3 principal components while retaining variance |
| **Real-time Customer Prediction** | Enter any customer's attributes and instantly get their predicted persona |
| **Segment Deep-Dive** | Filter by persona, view distribution histograms, download filtered CSV |
| **Correlation Heatmap** | Full feature correlation matrix on raw dataset — spots multicollinearity |
| **Streamlit Caching** | `@st.cache_resource` for model, `@st.cache_data` for DataFrames — zero reload lag |
| **Model Bundle Serialisation** | One pickle file bundles scaler + OHE + PCA + model + persona_map |
| **Reproducibility by Design** | `random_state=42` throughout; CONFIG dict for single-source-of-truth constants |

---

## 🏗️ Overall Architecture

```mermaid
graph TB
    subgraph Input["Input Layer"]
        CSV[("smartcart_customers.csv\n2,240 rows x 23 cols")]
    end

    subgraph Pipeline["Training Pipeline — seg-wise.ipynb"]
        EDA["EDA + Outlier Removal\nAge gt 90, Income gt 600K"]
        FE["Feature Engineering\nAge, Tenure, Total_Spending,\nTotal_Children, Living_With"]
        PP["Preprocessing\nOHE then StandardScaler"]
        PCA_STEP["PCA\n17 features to 3 components"]
        EVAL["Model Evaluation\nElbow + Silhouette\nK-Means vs Agglomerative"]
        TRAIN["Best Model Training\nAgglomerative, K=4"]
        SAVE["Model Serialisation\nsegwise_model.pkl"]
    end

    subgraph Outputs["Artefact Layer"]
        PKL[("segwise_model.pkl\nscaler+pca+ohe+model+persona_map")]
        PRED[("cluster_predictions.csv\n2,240 rows + Cluster + Persona")]
        SUMMARY[("cluster_summary.csv\nMean stats per cluster")]
    end

    subgraph App["Streamlit Dashboard — app.py"]
        PG1["Page 1: Dashboard\n3D scatter, KPI cards, donut, income bar"]
        PG2["Page 2: Cluster Explorer\nFilter + stats + download CSV"]
        PG3["Page 3: Predict Customer\nForm to transform to predict"]
        PG4["Page 4: Data Insights\nCorrelation heatmap + raw data"]
    end

    CSV --> EDA
    EDA --> FE --> PP --> PCA_STEP --> EVAL --> TRAIN --> SAVE
    TRAIN --> PRED
    TRAIN --> SUMMARY
    SAVE --> PKL
    PKL --> App
    PRED --> PG1
    PRED --> PG2
    SUMMARY --> PG1
```

---

## ⚙️ System Architecture

```mermaid
graph LR
    subgraph User["User"]
        BROWSER["Web Browser"]
    end

    subgraph Streamlit["Streamlit Server — app.py"]
        CACHE_R["cache_resource\nModel Bundle"]
        CACHE_D["cache_data\nDataFrames"]
        SIDEBAR["Sidebar Navigation\nRadio + Model Metadata"]
        PAGE_ROUTER["Page Router\nif elif blocks"]
        DASH["Dashboard Page"]
        CLUST["Cluster Explorer Page"]
        PRED_P["Predict Page\nOHE to Scale to PCA to Predict"]
        INSIGHT["Data Insights Page\nsns.heatmap + st.dataframe"]
    end

    subgraph Storage["File System"]
        PKL2[("models/segwise_model.pkl")]
        CSV2[("data/smartcart_customers.csv")]
        PREDS[("outputs/cluster_predictions.csv")]
        SUM[("outputs/cluster_summary.csv")]
    end

    subgraph ML["Model Bundle"]
        SC["StandardScaler"]
        OHE2["OneHotEncoder"]
        PCA2["PCA 3 components"]
        MODEL["AgglomerativeClustering or KMeans"]
        PMAP["persona_map dict"]
    end

    BROWSER -->|"HTTP GET"| Streamlit
    PKL2 -->|"pickle.load"| CACHE_R
    CSV2 -->|"pd.read_csv"| CACHE_D
    PREDS -->|"pd.read_csv"| CACHE_D
    CACHE_R --> ML
    CACHE_D --> PAGE_ROUTER
    SIDEBAR --> PAGE_ROUTER
    PAGE_ROUTER --> DASH
    PAGE_ROUTER --> CLUST
    PAGE_ROUTER --> PRED_P
    PAGE_ROUTER --> INSIGHT
    ML --> PRED_P
    Streamlit -->|"HTML + JS"| BROWSER
```

---

## 🧰 Technology Stack — Complete Breakdown

### Core ML / Data Science Layer

| Technology | Version | Category | Purpose in Project | Why Chosen | Key Features Used |
|---|---|---|---|---|---|
| scikit-learn | ≥1.4.2 | ML Framework | KMeans, AgglomerativeClustering, StandardScaler, PCA, OneHotEncoder | Industry standard; consistent API | `fit_transform`, `transform`, `predict`, `silhouette_score` |
| pandas | ≥2.1.4 | Data Processing | CSV loading, feature engineering, groupby, describe | Tabular de facto standard | `read_csv`, `groupby`, `describe`, `isnull`, `concat` |
| numpy | ≥1.26.4 | Numerical Computing | Array ops, triu mask, vstack for Agglomerative workaround | Low-level array control | `triu`, `ones_like`, `vstack`, `number` dtype |
| kneed | ≥0.8.5 | ML Utility | Auto-detect elbow in WCSS vs K curve | Removes subjective visual picking | `KneeLocator` with concave + decreasing |

### Visualisation Layer

| Technology | Version | Category | Purpose in Project | Why Chosen | Key Features Used |
|---|---|---|---|---|---|
| plotly | ≥5.18.0 | Interactive Charts | 3D scatter, donut pie, horizontal bar, histogram | Interactive hover/zoom/rotate in browser | `px.scatter_3d`, `px.pie`, `px.bar`, `px.histogram` |
| matplotlib | ≥3.8.2 | Static Charts | Correlation heatmap base figure | seaborn builds on it; `st.pyplot` renders inline | `subplots`, `tight_layout`, `close` |
| seaborn | ≥0.13.2 | Statistical Charts | Heatmap on dashboard Page 4 | One-line heatmaps with aesthetic defaults | `heatmap` with `annot`, `cmap`, `mask`, `fmt` |

### Application Layer

| Technology | Version | Category | Purpose in Project | Why Chosen | Key Features Used |
|---|---|---|---|---|---|
| streamlit | ≥1.31.0 | Web Framework | Entire dashboard UI | Zero-HTML, Python-only, reactive execution | `@cache_resource`, `@cache_data`, `columns`, `form`, `metric`, `plotly_chart`, `download_button` |

### Persistence Layer

| Technology | Version | Category | Purpose in Project | Why Chosen | Key Features Used |
|---|---|---|---|---|---|
| pickle | stdlib | Serialisation | Serialise full model bundle to one `.pkl` | Zero-dependency; preserves fitted sklearn state | `pickle.dump`, `pickle.load`, binary mode |

### Dev Tooling

| Technology | Version | Category | Purpose in Project | Why Chosen | Key Features Used |
|---|---|---|---|---|---|
| jupyter | ≥1.0.0 | Notebook Runtime | Training pipeline; cell-by-cell EDA | Interactive experimentation, inline charts | Kernel, cell execution, markdown cells |
| uv venv | — | Virtual Env | Isolated Python environment at `.venv/` | Fast dependency resolution | `uv pip install` |

---

## 🔬 ML Pipeline

```mermaid
flowchart TD
    A[("smartcart_customers.csv\n2,240 rows, 23 cols")] --> B["Import Libraries"]
    B --> C["CONFIG\nn_clusters=4, pca_components=3\noutlier thresholds, random_state=42"]
    C --> D["EDA\nshape, dtypes, describe, nulls, pairplot"]
    D --> E["Data Cleaning\nFill Income nulls with median\nRemove outlier age and income rows"]
    E --> F["Feature Engineering\nAge, Total_Spending, Total_Children\nCustomer_Tenure_Days, Living_With, Education"]
    F --> G["Preprocessing\nOneHotEncoder on Education + Living_With\nStandardScaler on all numeric"]
    G --> H["PCA — 3 components\nSave explained variance plot"]
    H --> I["Optimal K Search\nWCSS for K=1..10\nKneeLocator finds elbow at K=4"]
    I --> J["Silhouette Scoring\nKMeans K=4 vs Agglomerative K=4\nBest model selected"]
    J --> K["Final Training\nFit winning model\nLabel all customers\nMap cluster IDs to persona names"]
    K --> L["Serialisation\npickle.dump bundle\n→ models/segwise_model.pkl\n→ cluster_predictions.csv\n→ cluster_summary.csv"]
```

**Persona mapping by cluster statistics:**

| Cluster | Persona | Avg Income | Avg Spend | Defining Traits |
|---|---|---|---|---|
| 0 | Casual Buyers | ~$37K | $166 | High children, frequent web visits, low purchase rate |
| 1 | VIP Shoppers | ~$71K | $1,167 | Low children, catalog buyers, high engagement |
| 2 | Deal Hunters | ~$70K | $1,193 | Low children, deal purchasers, moderate web visits |
| 3 | Dormant Users | ~$37K | $169 | High children, low engagement, low spend |

---

## 🔄 Request Lifecycle

### Lifecycle 1: Dashboard Page Load

```
1. BROWSER REQUEST
   └── User navigates to localhost:8501
       → Streamlit serves app.py from top

2. CACHE CHECK — Model Bundle
   └── @st.cache_resource → load_model()
       → First run: open("models/segwise_model.pkl", "rb")
       → pickle.load() → reconstructs scaler, pca, ohe, model, persona_map
       → Subsequent runs: returns cached object (zero disk I/O)

3. CACHE CHECK — DataFrames
   └── @st.cache_data → load_data() → pd.read_csv("data/smartcart_customers.csv")
   └── @st.cache_data → load_predictions() → pd.read_csv("outputs/cluster_predictions.csv")

4. SIDEBAR → page = "Dashboard"

5. DASHBOARD PAGE
   └── 4 KPI metrics from df_pred
   └── px.scatter_3d(df_pred, x=Income, y=Total_Spending, z=Age, color=Persona)
   └── seg_counts = df_pred.Persona.value_counts() → px.pie (donut)
   └── income_by_seg = df_pred.groupby(Persona)[Income].mean() → px.bar

6. RESPONSE → Streamlit serialises to WebSocket → Browser renders HTML + Plotly JS
```

### Lifecycle 2: Predict New Customer (Form Submit)

```
1. USER FILLS FORM → 13 input widgets inside st.form (batched — no re-run until submit)

2. SUBMIT CLICKED → submitted = True → script re-runs
   └── Build new_customer_dict → pd.DataFrame([dict]) → df_new [1 row]
   └── ohe.transform(cat_input)     ← uses EXISTING vocabulary from training
   └── pd.concat([df_new, cat_df])  ← combine numeric + OHE columns
   └── df_final = df_final[bundle["features"]]  ← align to training column order
   └── scaler.transform(df_final)   ← z-score with TRAINING mean/std (not this row)
   └── pca.transform(X_scaled)      ← project onto EXISTING eigenvectors
   └── model.predict(X_pca)         ← distance to centroid (KMeans)
                                    OR refit workaround (Agglomerative)
   └── persona = persona_map[cluster_id]

3. DISPLAY → st.success(persona) + st.info(marketing tip)

ERROR PATH: pkl not found → st.error() → st.stop() (clean halt, no broken UI)
```

---

## 📊 Data Flow

```
RAW CSV [2240 × 23]
  → pd.read_csv()
  → Cleaning: drop outliers, fill Income nulls
  → Feature Engineering: +Age, +Total_Spending, +Total_Children, +Tenure, +Living_With
  → OneHotEncoder (Education 3-cat, Living_With 2-cat) → 5 OHE columns
  → StandardScaler (z-score, all numeric) → 12 scaled columns
  → PCA(n=3) → X_pca [2216 × 3]
  ┌──────────────────────────────────────┐
  │  K-Means(K=4)  vs  Agglomerative(K=4) │
  │  Silhouette scored → best wins        │
  └──────────────────────────────────────┘
  → cluster_labels [0,1,2,3] → persona_map → Persona strings
  → cluster_predictions.csv + cluster_summary.csv
  → segwise_model.pkl (scaler+ohe+pca+model+map bundled)
       ↓
  Streamlit @cache_resource loads pkl once
  Streamlit @cache_data loads CSVs once
       ↓
  Page 1: df_pred → Plotly 3D scatter + donut + bar
  Page 2: df_pred.filter(Persona) → histograms + CSV download
  Page 3: form → ohe.transform → scaler.transform → pca.transform → model.predict
  Page 4: df_raw → sns.heatmap + st.dataframe
```

---

<details>
<summary>📐 UML Diagrams Suite (click to expand)</summary>

### 1. Use Case Diagram

```mermaid
graph TD
    U1["Data Scientist"]
    U2["Business Analyst"]
    U3["Marketing Manager"]

    UC1["Run Training Notebook"]
    UC2["View Dashboard"]
    UC3["Explore Customer Segments"]
    UC4["Predict New Customer Segment"]
    UC5["Download Segment CSV"]
    UC6["View Correlation Heatmap"]
    UC7["Configure n_clusters"]

    U1 --> UC1
    U1 --> UC7
    U2 --> UC2
    U2 --> UC6
    U2 --> UC3
    U3 --> UC3
    U3 --> UC4
    U3 --> UC5
```

### 2. Class Diagram

```mermaid
graph TD
    subgraph NotebookPipeline["Notebook Pipeline"]
        CONFIG["CONFIG dict\n+data_path\n+n_clusters=4\n+pca_components=3\n+random_state=42"]
        FeatureEngineer["FeatureEngineer\n+engineer(df): df\n+encode_cat(df): arr\n+scale(df): arr"]
        ModelSelector["ModelSelector\n+elbow_search(X): int\n+silhouette_compare(X): model\n+label(model, X): arr"]
        BundleSaver["BundleSaver\n+save(path, bundle): void"]
    end

    subgraph Bundle["Model Bundle dict"]
        ScalerObj["StandardScaler\n+transform(X)"]
        OHEObj["OneHotEncoder\n+transform(X)"]
        PCAObj["PCA\n+transform(X)"]
        ModelObj["KMeans OR Agglomerative\n+predict(X)"]
        PersonaMap["persona_map dict\n{int: str}"]
    end

    subgraph StreamlitApp["Streamlit App"]
        AppLoader["AppLoader\n+load_model(): dict\n+load_data(): df\n+load_predictions(): df"]
        DashPage["DashboardPage\n+render_kpis()\n+render_3d()\n+render_donut()"]
        PredictPage["PredictPage\n+build_features(form): df\n+encode(ohe, df): df\n+predict(model, arr): int"]
    end

    CONFIG --> FeatureEngineer
    FeatureEngineer --> ModelSelector
    ModelSelector --> BundleSaver
    BundleSaver --> Bundle
    Bundle --> AppLoader
    AppLoader --> DashPage
    AppLoader --> PredictPage
```

### 3. Sequence Diagram — Prediction Flow

```mermaid
sequenceDiagram
    actor User
    participant Form as Prediction Form
    participant App as app.py
    participant OHE as OneHotEncoder
    participant Scaler as StandardScaler
    participant PCA as PCA
    participant Model as Clustering Model
    participant UI as Streamlit UI

    User->>Form: Fill 13 customer attributes
    User->>Form: Click Predict Segment
    Form->>App: submitted = True
    App->>App: Build new_customer_dict
    App->>OHE: transform(cat_input)
    OHE-->>App: encoded_cats [1x5]
    App->>App: concat numeric + cat then align columns
    App->>Scaler: transform(df_final)
    Scaler-->>App: X_scaled [1x17]
    App->>PCA: transform(X_scaled)
    PCA-->>App: X_pca [1x3]
    App->>Model: predict(X_pca)
    Model-->>App: cluster_id int
    App->>App: persona = persona_map[cluster_id]
    App->>UI: st.success(persona) + st.info(tip)
    UI-->>User: Display prediction and strategy
```

### 4. Activity Diagram — Training Pipeline

```mermaid
graph TD
    START(("START")) --> LOAD["Load CSV"]
    LOAD --> EDA["EDA: shape, types, nulls, describe"]
    EDA --> CLEAN["Clean: fill Income nulls, remove outliers"]
    CLEAN --> FE["Feature Engineering"]
    FE --> ENCODE["OneHotEncoder on categorical cols"]
    ENCODE --> SCALE["StandardScaler on numeric cols"]
    SCALE --> PCA_ACT["PCA: 17 to 3 components"]
    PCA_ACT --> ELBOW["Elbow search K=1..10 via KneeLocator"]
    ELBOW --> SIL["Silhouette: KMeans vs Agglomerative"]
    SIL --> BEST{Best Silhouette?}
    BEST -->|KMeans wins| TRAIN_KM["Train KMeans K=4"]
    BEST -->|Agglomerative wins| TRAIN_AG["Train AgglomerativeClustering K=4"]
    TRAIN_KM --> LABEL["Assign labels and persona names"]
    TRAIN_AG --> LABEL
    LABEL --> SAVE2["Save pkl + CSVs to disk"]
    SAVE2 --> END(("END"))
```

### 5. State Diagram — App States

```mermaid
stateDiagram-v2
    [*] --> Loading: App starts
    Loading --> BundleReady: pkl loaded and cached
    BundleReady --> Dashboard: User selects Dashboard
    BundleReady --> ClusterExplorer: User selects Explorer
    BundleReady --> PredictCustomer: User selects Predict
    BundleReady --> DataInsights: User selects Insights
    Dashboard --> BundleReady: User changes page
    ClusterExplorer --> BundleReady: User changes page
    PredictCustomer --> FormFilling: User fills inputs
    FormFilling --> Predicting: User submits form
    Predicting --> ResultShown: Prediction computed
    ResultShown --> FormFilling: User resets form
    Loading --> Error: pkl not found
    Error --> [*]: st.stop()
```

### 6. Component Diagram

```mermaid
graph LR
    subgraph CMP_NB["Training Notebook"]
        CMP1A["Data Loader"]
        CMP2A["Feature Engineer"]
        CMP3A["Preprocessor OHE + Scaler"]
        CMP4A["PCA Reducer"]
        CMP5A["Model Evaluator"]
        CMP6A["Artefact Writer"]
    end

    subgraph CMP_APP["Streamlit App"]
        CMP1B["Cache Manager"]
        CMP2B["Page Router"]
        CMP3B["Dashboard Page"]
        CMP4B["Explorer Page"]
        CMP5B["Predictor Page"]
        CMP6B["Insights Page"]
    end

    subgraph CMP_STORE["Storage"]
        CMP1C[("segwise_model.pkl")]
        CMP2C[("cluster_predictions.csv")]
        CMP3C[("smartcart_customers.csv")]
    end

    CMP6A --> CMP1C
    CMP6A --> CMP2C
    CMP1C --> CMP1B
    CMP2C --> CMP1B
    CMP3C --> CMP1B
    CMP1B --> CMP2B
    CMP2B --> CMP3B
    CMP2B --> CMP4B
    CMP2B --> CMP5B
    CMP2B --> CMP6B
```

### 7. Deployment Diagram

```mermaid
graph TD
    subgraph DEV["Developer Machine"]
        VENV[".venv uv virtualenv"]
        NB["Jupyter Notebook — Training Pipeline"]
        ST["Streamlit Server localhost:8501"]
        FS["File System D:/AIML/projects/segwise/"]
    end

    subgraph CLOUD["Cloud Deployment Options"]
        HF["Hugging Face Spaces"]
        GH["GitHub Repo"]
    end

    VENV --> NB
    VENV --> ST
    NB -->|"writes artefacts"| FS
    ST -->|"reads artefacts"| FS
    FS -->|"git push"| GH
    GH -->|"deploy"| HF
```

### 8. Object Diagram — Model Bundle

```mermaid
graph TD
    BUNDLE["bundle: dict"]
    SC_OBJ["scaler: StandardScaler\nmean_ and scale_ from training data"]
    PCA_OBJ["pca: PCA\nn_components_=3\nexplained_variance_ratio_ stored"]
    OHE_OBJ["ohe: OneHotEncoder\ncategories_ for Education and Living_With\nhandle_unknown=ignore"]
    MODEL_OBJ["model: AgglomerativeClustering\nn_clusters=4, linkage=ward\nlabels_ from training"]
    PMAP_OBJ["persona_map: dict\n0: Casual Buyers\n1: VIP Shoppers\n2: Deal Hunters\n3: Dormant Users"]
    META["n_clusters: 4\nbest_name: Agglomerative\nfeatures: training column list"]

    BUNDLE --> SC_OBJ
    BUNDLE --> PCA_OBJ
    BUNDLE --> OHE_OBJ
    BUNDLE --> MODEL_OBJ
    BUNDLE --> PMAP_OBJ
    BUNDLE --> META
```

### 9. Package Diagram

```mermaid
graph TD
    PKG_APP["app.py\nStreamlit Dashboard"]
    PKG_NB["notebooks/seg-wise.ipynb\nTraining Pipeline"]
    PKG_DATA["data/\nsmartcart_customers.csv"]
    PKG_MODELS["models/\nsegwise_model.pkl"]
    PKG_SKLEARN["scikit-learn\nKMeans, Agglomerative, Scaler, PCA, OHE"]
    PKG_PLOTLY["plotly + seaborn + matplotlib"]
    PKG_PANDAS["pandas + numpy"]
    PKG_STREAMLIT["streamlit"]

    PKG_NB --> PKG_DATA
    PKG_NB --> PKG_SKLEARN
    PKG_NB --> PKG_PANDAS
    PKG_NB --> PKG_MODELS
    PKG_APP --> PKG_MODELS
    PKG_APP --> PKG_DATA
    PKG_APP --> PKG_STREAMLIT
    PKG_APP --> PKG_PLOTLY
    PKG_APP --> PKG_PANDAS
```

### Swimlane

```mermaid
sequenceDiagram
    participant DS as Data Scientist
    participant NB as Jupyter Notebook
    participant FS as File System
    participant ST as Streamlit App
    participant BA as Business Analyst

    DS->>NB: Run all cells
    NB->>FS: Read smartcart_customers.csv
    FS-->>NB: 2240 rows
    NB->>NB: EDA + Clean + Feature Engineer + OHE + Scale + PCA + Train
    NB->>FS: Write segwise_model.pkl + CSVs
    DS->>ST: streamlit run app.py
    ST->>FS: load pkl + CSVs (cached)
    FS-->>ST: Bundle + DataFrames
    BA->>ST: Open localhost:8501
    ST-->>BA: Dashboard with 3D scatter
    BA->>ST: Predict New Customer form submit
    ST->>ST: OHE then Scale then PCA then Predict
    ST-->>BA: Persona + marketing strategy
```

</details>

---

<details>
<summary>📊 Data Flow Diagrams (click to expand)</summary>

### DFD Level 0 — Context Diagram

```mermaid
graph LR
    E1["E1 Data Scientist"]
    E2["E2 Business Analyst"]
    E3["E3 Marketing Manager"]
    P0(("0\nSegWise\nPlatform"))
    E1 -->|"raw CSV + config"| P0
    P0 -->|"trained model + plots"| E1
    E2 -->|"dashboard query"| P0
    P0 -->|"segment insights"| E2
    E3 -->|"new customer attributes"| P0
    P0 -->|"segment prediction + strategy"| E3
```

### DFD Level 1 — System Decomposition

```mermaid
graph TD
    E1B["E1 Data Scientist"]
    E2B["E2 Business Analyst"]
    E3B["E3 Marketing Manager"]

    P1(("1.0\nIngest and Validate\nRaw Data"))
    P2(("2.0\nEngineer Features\nand Preprocess"))
    P3(("3.0\nTrain and Select\nBest Model"))
    P4(("4.0\nSerialise Model\nand Artefacts"))
    P5(("5.0\nServe Dashboard\nand Predictions"))

    D1[("D1: Raw Customer Store\nsmartcart_customers.csv")]
    D2[("D2: Processed Feature Store\nin-memory DataFrame")]
    D3[("D3: Model Store\nsegwise_model.pkl")]
    D4[("D4: Prediction Store\ncluster_predictions.csv")]

    E1B -->|"CSV upload"| P1
    P1 -->|"cleaned data"| D1
    D1 -->|"raw rows"| P2
    P2 -->|"scaled features"| D2
    D2 -->|"X_pca"| P3
    P3 -->|"fitted model + labels"| D3
    P3 -->|"cluster assignments"| D4
    D3 -->|"model bundle"| P4
    P4 -->|"pkl + CSVs"| E1B
    D3 -->|"load bundle"| P5
    D4 -->|"predictions"| P5
    E2B -->|"page navigation"| P5
    E3B -->|"customer form data"| P5
    P5 -->|"segment visualisations"| E2B
    P5 -->|"persona + strategy"| E3B
```

</details>

---

## 📁 Folder Structure

```
segwise/
├── app.py                          ← Streamlit dashboard (1073 lines, fully annotated)
├── requirements.txt                ← Python dependencies
├── notebooks/
│   └── seg-wise.ipynb              ← Main training pipeline (27 cells)
├── segwise_eda.ipynb               ← Standalone EDA exploration notebook
├── data/
│   └── smartcart_customers.csv     ← Raw dataset (2,240 rows × 23 columns)
├── models/
│   └── segwise_model.pkl           ← Serialised model bundle [run notebook first]
├── outputs/
│   ├── 00_pairplot_raw.png
│   ├── 00b_correlation_heatmap.png
│   ├── 01_pca_raw.png
│   ├── 02_optimal_k.png
│   ├── 03_pca_3d_clusters.png
│   ├── 04_income_vs_spending.png
│   ├── 05_cluster_sizes.png
│   ├── 06_cluster_heatmap.png
│   ├── cluster_predictions.csv     ← All customers + Cluster + Persona
│   └── cluster_summary.csv         ← Mean stats per cluster
├── .venv/                          ← uv virtual environment (not committed)
├── .vscode/settings.json           ← Python interpreter → .venv
└── .gitignore
```

---

## 📦 Dataset

**SmartCart Customer Dataset** — `data/smartcart_customers.csv` — 2,240 rows × 23 columns

| Column | Type | Description |
|--------|------|-------------|
| `ID` | int | Customer ID (dropped after load) |
| `Year_Birth` | int | → engineered to `Age` |
| `Education` | str | Graduation/Master/PhD → simplified to 3 levels |
| `Marital_Status` | str | → `Living_With` (Partner / Alone) |
| `Income` | float | Annual household income (24 nulls → filled with median) |
| `Kidhome` | int | Kids in household |
| `Teenhome` | int | Teens in household |
| `Dt_Customer` | str | Enrolment date → `Customer_Tenure_Days` |
| `Recency` | int | Days since last purchase |
| `MntWines` .. `MntGoldProds` | int | Spend per category → summed to `Total_Spending` |
| `NumDealsPurchases` | int | Discount purchases |
| `NumWebPurchases` | int | Web-channel purchases |
| `NumCatalogPurchases` | int | Catalog purchases |
| `NumStorePurchases` | int | In-store purchases |
| `NumWebVisitsMonth` | int | Non-purchase website visits |
| `Complain` | int | Complained in last 2 years (0/1) |
| `Response` | int | Accepted last campaign offer (0/1) |

---

## ✅ Prerequisites

- Python 3.11+
- `uv` package manager (recommended) or `pip`
- Modern browser for Streamlit

---

## 🚀 Installation

```bash
# Clone
git clone https://github.com/KARTHIKAKRISHNA123/seg-wise.git
cd seg-wise

# Create venv
uv venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
uv pip install -r requirements.txt

# MANDATORY: Run the training notebook first
# Open notebooks/seg-wise.ipynb → Run All Cells (1 → 27)
# Creates: models/segwise_model.pkl + outputs/cluster_predictions.csv
```

---

## ▶️ Running the App

```bash
# From project root
cd D:\AIML\projects\segwise
streamlit run app.py
# → Open http://localhost:8501
```

> The app shows a red error box and stops if `models/segwise_model.pkl` is missing. Always run the notebook first.

---

## 📄 Page Reference

| Page | Nav | Description |
|------|-----|-------------|
| Dashboard | Sidebar → Dashboard | 4 KPI cards, 3D scatter, donut pie, avg income bar |
| Cluster Explorer | Sidebar → Cluster Explorer | Select persona → stats table → histograms → CSV download |
| Predict New Customer | Sidebar → Predict New Customer | 13-field form → real-time persona + marketing tip |
| Data Insights | Sidebar → Data Insights | Raw data overview, null counts, Pearson heatmap, data preview |

---

## 🗄️ Data Schema

### `cluster_predictions.csv`
All training customers + their assigned cluster:
`Income, Recency, NumDealsPurchases, ..., Total_Spending, Education_Graduate, ..., Living_With_Partner, Cluster (int), Persona (str)`

### `cluster_summary.csv`
Per-cluster mean statistics:
`Cluster | Income | Total_Spending | Age | Recency | NumDealsPurchases | NumWebVisitsMonth | Total_Children | Complain`

---

## 🧠 Engineering Decisions & Tradeoffs

| Decision | Choice | Alternative | Reason |
|---|---|---|---|
| Clustering | Both KMeans + Agglomerative; best silhouette wins | Only KMeans | Validates choice; Agglomerative handles non-spherical shapes |
| Dim reduction | PCA 3 components | No PCA / t-SNE | 3D visualisable; t-SNE non-deterministic and non-invertible |
| Optimal K | KneeLocator + silhouette | Manual selection | Reproducible, data-driven |
| Model persistence | Single pickle bundle | Separate files | Atomic load; no partial-load bugs |
| Caching | `@cache_resource` + `@cache_data` | No cache | Without it: pkl reloads on every widget interaction |
| Form batching | `st.form` for all 13 inputs | Individual inputs | 1 re-run on submit vs 13 re-runs while filling |
| Scaling | StandardScaler z-score | MinMaxScaler | Handles outliers better; k-means is distance-sensitive |

---

## 🔐 Security Considerations

- No authentication — local analysis tool, single-user
- All inference fully offline — no data leaves the machine
- SmartCart CSV contains income data — keep `data/` in `.gitignore`
- Only load `.pkl` files from your own notebook — `pickle.load` executes arbitrary code

---

## ⚡ Performance Optimizations

| Optimization | Location | Impact |
|---|---|---|
| `@st.cache_resource` on model load | app.py | Model loads once; all pages share same scaler/pca/ohe in memory |
| `@st.cache_data` on CSV loads | app.py | DataFrames loaded once per session |
| `st.form` for prediction inputs | Page 3 | 13 widget inputs batched into 1 script re-run |
| `plt.close(fig)` after `st.pyplot` | Page 4 | Prevents matplotlib memory leak across re-runs |

---

## 🛠️ Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: models/segwise_model.pkl` | Notebook not run | Run all cells in `notebooks/seg-wise.ipynb` |
| `KeyError: 'Total_Spending'` | Feature mismatch | Delete pkl → re-run notebook from Cell 1 |
| Plotly charts blank (no data points) | DataFrame column name mismatch or dtype issue | Add `st.write(df_pred.dtypes)` before the chart call; ensure column names exactly match those saved in pkl |
| `Prediction failed` on Page 3 | Column order mismatch | Check `bundle["features"]` vs `new_customer_dict` keys |
| `ModuleNotFoundError` | Packages not in venv | `pip install -r requirements.txt` inside activated `.venv` |

---

## 🤝 Contributing

```bash
git checkout -b feature/your-feature
# Make changes — ensure notebook runs clean (restart kernel → run all)
# Ensure app starts without errors: streamlit run app.py
git commit -m "feat: description"
git push origin feature/your-feature
# Open Pull Request
```

---

## 👩‍💻 Author

<div align="center">

**Karthika Krishna M**

Final-Year B.E. Computer Science & Engineering  
Anna University Regional Campus, Tirunelveli | Class of 2026

[![GitHub](https://img.shields.io/badge/GitHub-KARTHIKAKRISHNA123-181717?logo=github)](https://github.com/KARTHIKAKRISHNA123)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-KarthikaKrishna123-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/KarthikaKrishna123)

</div>

---

<div align="center">

© 2025 SegWise · Built by Karthika Krishna M · Unsupervised ML · PCA · Agglomerative Clustering · Streamlit

</div>
