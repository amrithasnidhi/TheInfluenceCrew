# 📊 The Influence Crew — Business Analytics Project
### Analyzing the Impact of Influencer Content on Consumer Purchase Decisions

---

## 🗂️ Project Files

| File | Description |
|------|-------------|
| `influencer_analysis.ipynb` | Complete Jupyter Notebook — all 5 BA methodologies |
| `streamlit_dashboard.py` | Live interactive Streamlit dashboard |
| `requirements.txt` | Python dependencies |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your data file
Copy the Excel survey file and rename it:
```
influencer_survey.xlsx
```
Place it in the **same folder** as `streamlit_dashboard.py`.

### 3. Run the Streamlit dashboard
```bash
streamlit run streamlit_dashboard.py
```
Opens at: http://localhost:8501

### 4. Open the Jupyter Notebook
```bash
jupyter notebook influencer_analysis.ipynb
```
> Inside the notebook, update the file path in Cell 3 if needed:
> `df_raw = pd.read_excel('influencer_survey.xlsx')`

---

## 📋 Methodologies Covered

| # | Method | Implementation |
|---|--------|---------------|
| 1 | Business Problem Structuring & KPI Mapping | Notebook Section 1, Dashboard Tab 6 |
| 2 | EDA & Correlation Diagnostics | Notebook Sections 3-4, Dashboard Tabs 1-2 |
| 3 | Dimensionality Reduction (PCA) | Notebook Section 5, Dashboard Tab 5 |
| 4 | Predictive & Causal Modeling | Notebook Section 6, Dashboard Tab 3 |
| 5 | Association Rule Mining | Notebook Section 7, Dashboard Tab 4 |

---

## 📊 Dashboard Features

- **Live Sidebar Filters** — Filter by age, gender, spending, trust driver, platform
- **Tab 1: Overview** — KPIs, demographics, platform reach
- **Tab 2: Factor Analysis** — Live correlation explorer (any factor vs any outcome)
- **Tab 3: Predictions** — Logistic regression, Random Forest, ROC, coefficients
- **Tab 4: Pattern Mining** — Association rules with adjustable thresholds
- **Tab 5: Trends & PCA** — Scree plot, loadings, consumer segment map
- **Tab 6: Recommendations** — 5 strategic recommendations (auto-updates with filters)

---

## 👥 Team — Group 5 The Influence Crew

- Amritha S Nidhi (CB.SC.U4CSE23404)
- Deeptha Kannoth Padinjareil (CB.SC.U4CSE23015)
- Yuvan R (CB.SC.U4CSE23344)
- Nanthakumaran A (CB.SC.U4CSE23341)
- Sushil V (CB.SC.U4CSE23248)
