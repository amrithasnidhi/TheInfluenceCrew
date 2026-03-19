# 🛠️ Developer Documentation: The Influence Crew Dashboard

This document provides technical details, architecture overview, and setup instructions for the developers maintaining or extending **The Influence Crew** Business Analytics Dashboard.

## 📌 Project Overview
The project is a Streamlit-based interactive web application that analyzes the impact of influencer content on consumer purchase decisions. It features a live filtering system, KPI overviews, live factor analysis (Spearman correlation), predictive modeling (Logistic Regression, Random Forest, Linear Regression), pattern mining (Association Rules), Social Network Analysis (SNA), and dimensionality reduction (PCA).

### Key Files
- `streamlit_dashboard.py`: The main entry point for the Streamlit web application. Contains all UI components, data processing, machine learning models, and visualizations.
- `influencer_analysis.ipynb`: A Jupyter Notebook containing the exploratory data analysis (EDA) and model prototyping.
- `requirements.txt`: Lists all Python dependencies required to run the application.

## 🏗️ Architecture & Tech Stack
- **Framework:** Streamlit (Frontend & Backend integration)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (PCA, Logistic/Linear Regression, Random Forest, metrics)
- **Statistics:** SciPy (Spearman correlation, Chi-Square)
- **Network Analysis:** NetworkX (Directed Acyclic Graphs, Bipartite networks, Centrality measures)
- **Visualization:** Matplotlib, Seaborn

## ⚙️ Data Flow & Preprocessing
The application loads data using the `@st.cache_data` decorated `load_data()` function which looks for `influencer_survey.xlsx`. If the file is not found, a synthetic fallback dataset is generated using NumPy to ensure the dashboard remains functional for demonstration purposes.

The `preprocess(df_raw)` function handles:
- **Column Renaming:** Maps verbose survey questions to concise variable names.
- **Ordinal Encoding:** Converts categorical variables (e.g., spending, daily time, engagement) into numerical ordinal values using predefined mappings.
- **Binary Features:** Creates binary flags like `high_ad_responsive` and `made_purchase` for classification tasks.
- **One-Hot Encoding:** Extracts platform usage and purchase motivations into individual binary columns.

## 🧩 Dashboard Modules (Tabs)
The dashboard is structurally divided into several tabs, each handling a specific analytical methodology:

1. **📊 Overview:** Displays high-level KPIs and demographic breakdowns using Matplotlib pie/bar charts.
2. **🔗 Factor Analysis:** Calculates Spearman correlation dynamically based on user-selected X and Y axes from the sidebar. Generates a correlation heatmap.
3. **🤖 Predictions:** Implements a Logistic Regression model to predict `high_ad_responsive` status, displaying ROC curves and Confusion Matrices. Uses Random Forest for feature importance and Linear Regression for predicting purchase frequency.
4. **⛏️ Pattern Mining:** Manual implementation of logic to identify Association Rules (Antecedents -> Consequent) based on user-adjusted confidence and support thresholds, calculating Lift.
5. **🕸️ Social Network Analysis:** Uses NetworkX to build and render different types of networks:
    - Platform Co-Occurrence (Graph)
    - Trust ↔ Motivation (Bipartite Graph)
    - Content Co-Occurrence (Graph)
    - Influence Flow (Directed Acyclic Graph)
6. **📈 Trends & PCA:** Performs Principal Component Analysis (StandardScaler + PCA) to segment consumers and visualize variance (Scree plot, Loadings heatmap, PC1 vs PC2 scatter). Includes Cohort Trend Simulations.
7. **💰 ROI Simulator:** Calculates estimated reach, clicks, conversions, and ROI using predefined benchmarks based on influencer type and platform multipliers.

## 🎨 UI & Styling
Custom CSS is injected via `st.markdown(..., unsafe_allow_html=True)` to control the application's visual theme (Dark Mode, specific fonts like 'Syne' and 'DM Sans', custom metrics boxes, and hero titles). Matplotlib parameter updates (`plt.rcParams.update`) ensure that all generated charts match the dark theme aesthetics of the Streamlit app.

## 🛠️ Local Setup & Execution
1. Ensure Python 3.9+ is installed.
2. Install dependencies: `pip install -r requirements.txt`
3. Place `influencer_survey.xlsx` in the root directory.
4. Run the app: `streamlit run streamlit_dashboard.py`

## 🚀 Further Development
- **Modularity:** Consider refactoring `streamlit_dashboard.py` by splitting the tabs and complex logic (like SNA or PCA) into separate modules/files (e.g., `utils.py`, `models.py`) to improve maintainability.
- **Caching:** Ensure `@st.cache_data` is used judiciously on heavy operations outside of just data loading to optimize performance during live filter changes.
