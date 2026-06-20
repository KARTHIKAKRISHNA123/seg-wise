---
title: SegWise Customer Intelligence
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# SegWise — Customer Intelligence Platform

SegWise is an end-to-end unsupervised machine learning pipeline and interactive web dashboard. It segments customer data to provide actionable business intelligence and marketing ROI.

Built by: Karthika Krishna M

## Architecture & Tech Stack

This project leverages a traditional, robust data science workflow, utilizing industry-standard libraries to prevent technical debt:

* **Frontend/UI:** Streamlit
* **Machine Learning:** scikit-learn (K-Means, Agglomerative Clustering, PCA)
* **Data Processing:** pandas, NumPy
* **Visualisation:** Plotly, Seaborn, Matplotlib
* **Serialization:** pickle

## Repository Structure

The codebase serves as the single source of truth for the entire pipeline:

* `app.py`: The Streamlit dashboard application and deployment entry point.
* `notebooks/SegWise.ipynb`: The core ML pipeline (EDA, Feature Engineering, Training, Serialization).
* `models/segwise_model.pkl`: The serialized model bundle containing the scaler, PCA, OneHotEncoder, and clustering algorithm.
* `data/smartcart_customers.csv`: The raw, read-only dataset.
* `outputs/`: Auto-generated insights, charts, and cluster prediction logs.

## Local Execution

To run this application in a local sandbox environment:

1. Clone the repository and initialize Git LFS to pull the large binaries.
2. Activate your virtual environment.
3. Install dependencies:
```bash
   uv pip install -r requirements.txt