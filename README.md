# ✨ Luminary

Luminary is a data-driven study into the world of influencer marketing — exploring how social media content shapes consumer trust, drives purchase decisions, and varies across age groups, platforms, and ad formats. Built on survey data from 4,210 respondents, it combines statistical modeling, machine learning, and interactive visualization to surface insights that help brands invest smarter in the creator economy.

---

## 🗂️ Project Files

| File | Description |
|------|-------------|
| `influencer_analysis.ipynb` | Complete Jupyter Notebook — all 5 BA methodologies |
| `streamlit_dashboard.py` | Live interactive Streamlit dashboard |
| `requirements.txt` | Python dependencies |
| `influencer_survey.xlsx` | Project Dataset |

---

## 🌐 Deployment

| Platform | Link |
|----------|------|
| **📊 Interactive Dashboard** (Hugging Face) | [https://huggingface.co/spaces/deep11122/ba](https://huggingface.co/spaces/deep11122/ba) |

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
| 1 | Business Problem Structuring & KPI Mapping | Notebook Section 1, Dashboard Tab 2 |
| 2 | EDA & Correlation Diagnostics | Notebook Sections 3-4, Dashboard Tabs 2-3 |
| 3 | Dimensionality Reduction (PCA) | Notebook Section 5, Dashboard Tab 7 |
| 4 | Predictive & Causal Modeling | Notebook Section 6, Dashboard Tab 4 |
| 5 | Association Rule Mining | Notebook Section 7, Dashboard Tab 5 |

---

## 📊 Dashboard Tabs

| # | Tab |
|---|-----|
| 1 | **Intro** |
| 2 | **Overview** — KPIs, demographics, platform reach |
| 3 | **Factor Analysis** — Live correlation explorer |
| 4 | **Predictions** — Logistic regression, Random Forest, ROC |
| 5 | **Pattern Mining** — Association rules with adjustable thresholds |
| 6 | **Network Analysis** — Relationship mapping across variables |
| 7 | **Trends & PCA** — Scree plot, loadings, consumer segment map |
| 8 | **Recommendations** — Strategic recommendations (auto-updates with filters) |
| 9 | **Clustering** — Consumer segment clustering |
| 10 | **ROI Simulator** — Budget allocation and return simulation |
| 11 | **Text Analytics** — Sentiment and keyword analysis |
| 12 | **Export** — Download filtered data and reports |
| 13 | **Conclusion** — Summary of findings and next steps |

---

## 📈 Tableau Story — Luminary Insights

Our Tableau story presents two chapters of the analysis:

**Chapter 1 — Current Audience Demographics & ROI Analysis**
- Marketing ROI Heatmap across ad formats and purchase frequency
- Audience Age Demographic breakdown (4,210 total respondents)
- Core Trust Factors ranked by respondent count
- Instagram Sales Conversion by user type

**Chapter 2 — Predictive Insights: Sales Lift & Trust Factor Projections**
- Projected Lift by Ad Format (Current vs Projected Future Purchases)
- Projected Sales Lift by Age Group
- Projected Lift by Platform (Instagram users vs non-users)
- Projected Sales Lift by Trust Factor (bubble chart)

**✨ Data Agent Feature**
The Tableau story includes a built-in **Data Agent dropdown** — a curated set of preset questions embedded directly in the dashboard. Users select a question from the dropdown (e.g. *"Which ad format drives the most purchases?"*) and Tableau's Data Agent instantly surfaces a data-backed answer from the underlying survey data — making insights accessible without any manual exploration.

---

## 👥 Team — Group 5 · Luminary

- Amritha S Nidhi (CB.SC.U4CSE23404)
- Deeptha Kannoth Padinjareil (CB.SC.U4CSE23015)
- Yuvan R (CB.SC.U4CSE23344)
- Nanthakumaran A (CB.SC.U4CSE23341)
- Sushil V (CB.SC.U4CSE23248)
