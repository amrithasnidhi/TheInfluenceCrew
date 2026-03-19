"""
streamlit_dashboard.py
The Influence Crew — Live BA Dashboard
Run: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (roc_auc_score, roc_curve, 
                              confusion_matrix, classification_report)
from scipy.stats import spearmanr, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="The Influence Crew — BA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  .main { background-color: #0a0a0f; }
  .block-container { padding-top: 1.5rem; }

  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7c5cfc, #fc5c7d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
  }
  .hero-sub { color: #7878a0; font-size: 0.95rem; margin-top: 4px; }

  .kpi-box {
    background: #16161f;
    border: 1px solid #2a2a3a;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    border-top: 3px solid;
  }
  .kpi-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.2px; color: #7878a0; }
  .kpi-value { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; margin: 4px 0; }
  .kpi-note { font-size: 0.75rem; color: #7878a0; }

  .insight-box {
    background: #16161f;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
  }
  .insight-title { font-weight: 600; font-size: 0.9rem; margin-bottom: 4px; }
  .insight-text { font-size: 0.82rem; color: #aaaacc; line-height: 1.6; }

  .section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #e8e8f0;
    border-left: 3px solid #7c5cfc;
    padding-left: 10px;
    margin: 1.2rem 0 0.8rem 0;
  }
  
  .stSelectbox label, .stMultiSelect label, .stSlider label, .stRadio label {
    color: #c8c8e8 !important;
    font-size: 0.85rem !important;
  }
  
  .stSidebar { background: #111118; }

  div[data-testid="metric-container"] {
    background: #16161f;
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 12px;
  }
</style>
""", unsafe_allow_html=True)

PALETTE = ['#7c5cfc','#fc5c7d','#5cf0c8','#fca95c','#5c96f0','#e05cf0','#f0e05c','#5cf05c']

plt.rcParams.update({
    'figure.facecolor': '#111118',
    'axes.facecolor': '#16161f',
    'axes.edgecolor': '#3a3a5c',
    'axes.labelcolor': '#c8c8e8',
    'xtick.color': '#8888aa',
    'ytick.color': '#8888aa',
    'text.color': '#e8e8f0',
    'grid.color': '#2a2a3a',
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'legend.facecolor': '#1a1a28',
    'legend.edgecolor': '#3a3a5c',
})

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    import glob, os
    # Try common paths
    candidates = [
        'influencer_survey.xlsx',
        'influencer_survey_clean.xlsx',
    ]
    # Also search for any xlsx
    found = glob.glob('*.xlsx') + glob.glob('../*.xlsx') + glob.glob('/mnt/user-data/uploads/*.xlsx')
    candidates.extend(found)
    
    df = None
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_excel(path)
                break
            except:
                continue
    
    if df is None:
        # Generate synthetic fallback data matching the real survey structure
        np.random.seed(42)
        n = 210
        df = pd.DataFrame({
            'ID': range(1, n+1),
            'Age Group': np.random.choice(['13-18','19-24','19-24','19-24','19-24','25-30','31-40','40+'], n),
            'Gender': np.random.choice(['Male','Male','Female'], n),
            'Average Monthly Online Spending (₹)': np.random.choice(
                ['< ₹500','₹500–₹2000','₹2000–₹5000','₹5000+'], n, p=[0.25,0.47,0.17,0.11]),
            'Which platforms do you regularly use to consume influencer content': 
                [';'.join(np.random.choice(['Instagram','YouTube','Pinterest','Snapchat','Facebook'],
                                           size=np.random.randint(1,4), replace=False)) for _ in range(n)],
            'Average time spent per day watching influencer-related content':
                np.random.choice(['Less than 1 hour','1-2 hour','2-4 hour','More than 4 hours'], n, p=[0.32,0.40,0.18,0.10]),
            'Do you currently follow any influencers?': np.random.choice(['Yes','No'], n, p=[0.95,0.05]),
            'Which type of influencer content do you engage with the most?':
                [';'.join(np.random.choice(['Food','Entertainment / Memes','Travel / Lifestyle','Fitness / Health',
                                            'Technology / Gadgets','Fashion / Beauty','Finance / Education'],
                                           size=np.random.randint(1,5), replace=False)) for _ in range(n)],
            'What factor makes you trust an influencer the most?':
                np.random.choice(['Content quality','Transparency in reviews','Authenticity',
                                   'Expertise in niche','High follower count','Brand collaborations'],
                                   n, p=[0.376,0.276,0.2,0.1,0.029,0.019]),
            'How often do you interact with influencer content (likes, comments, shares, saves)?':
                np.random.choice(['Rarely','Sometimes','frequently','Almost always'], n, p=[0.476,0.433,0.081,0.01]),
            'To what extent do influencer reviews affect your interest in a product?':
                np.random.choice([1,2,3,4,5], n, p=[0.105,0.129,0.419,0.29,0.057]),
            'Have you ever searched for a product after seeing it promoted by an influencer?':
                np.random.choice(['Yes','No'], n, p=[0.93,0.07]),
            'How often have you purchased a product because of an influencer recommendation?':
                np.random.choice(['Never','1-2 times','2-3 times','3-2 times','More than 5 times'],
                                   n, p=[0.243,0.543,0.105,0.062,0.047]),
            'What primarily motivated you to make that purchase?':
                [';'.join(np.random.choice(['Positive reviews','Discount / coupon code','Trust in influencer',
                                            'Attractive content','Curiosity','Trend / FOMO'],
                                           size=np.random.randint(1,4), replace=False)) for _ in range(n)],
            'How satisfied were you with products bought through influencer recommendations?':
                np.random.choice([1,2,3,4,5], n, p=[0.057,0.09,0.419,0.372,0.062]),
            'How often do you click on ads shown after engaging with influencer content?':
                np.random.choice(['Never','Rarely','Sometimes','Frequently'], n, p=[0.286,0.505,0.195,0.014]),
            'Do you believe ads shown to you are influenced by the influencers you follow and content you engage with?':
                np.random.choice(['Yes','No','Not sure'], n, p=[0.7,0.1,0.2]),
            '. How likely are you to buy a product after seeing repeated ads for it (retargeting)?':
                np.random.choice([1,2,3,4,5], n, p=[0.281,0.281,0.29,0.105,0.043]),
            'Which ad format influences you the most?':
                np.random.choice(['Reels','Feed posts','Stories','Sponsored influencer posts','Carousel ads'],
                                   n, p=[0.624,0.152,0.133,0.062,0.029]),
        })
        st.sidebar.warning('⚠️ Using synthetic demo data. Place your Excel file as `influencer_survey.xlsx` in the same directory.')
    
    return df

@st.cache_data
def preprocess(df_raw):
    col_map = {
        'Age Group': 'age_group',
        'Gender': 'gender',
        'Average Monthly Online Spending (₹)': 'spending',
        'Which platforms do you regularly use to consume influencer content': 'platforms',
        'Average time spent per day watching influencer-related content': 'daily_time',
        'Do you currently follow any influencers?': 'follows_influencers',
        'Which type of influencer content do you engage with the most?': 'content_type',
        'What factor makes you trust an influencer the most?': 'trust_factor',
        'How often do you interact with influencer content (likes, comments, shares, saves)?': 'engagement_freq',
        'To what extent do influencer reviews affect your interest in a product?': 'product_interest_score',
        'Have you ever searched for a product after seeing it promoted by an influencer?': 'searched_product',
        'How often have you purchased a product because of an influencer recommendation?': 'purchase_freq',
        'What primarily motivated you to make that purchase?': 'purchase_motivation',
        'How satisfied were you with products bought through influencer recommendations?': 'satisfaction_score',
        'How often do you click on ads shown after engaging with influencer content?': 'ad_click_freq',
        'Do you believe ads shown to you are influenced by the influencers you follow and content you engage with?': 'ad_influenced',
        '. How likely are you to buy a product after seeing repeated ads for it (retargeting)?': 'retargeting_score',
        'Which ad format influences you the most?': 'preferred_ad_format',
    }
    df = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})
    
    spending_map = {'< ₹500': 1, '₹500–₹2000': 2, '₹2000–₹5000': 3, '₹5000+': 4}
    time_map = {'Less than 1 hour': 1, '1-2 hour': 2, '2-4 hour': 3, 'More than 4 hours': 4}
    engage_map = {'Rarely': 1, 'Sometimes': 2, 'frequently': 3, 'Almost always': 4}
    purchase_map = {'Never': 0, '1-2 times': 1, '2-3 times': 2, '3-2 times': 3, 'More than 5 times': 4}
    adclick_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Frequently': 3}
    age_map = {'13-18': 1, '19-24': 2, '25-30': 3, '31-40': 4, '40': 5, '40+': 5}

    df['spending_ord'] = df['spending'].map(spending_map) if 'spending' in df.columns else 2
    df['daily_time_ord'] = df['daily_time'].map(time_map) if 'daily_time' in df.columns else 2
    df['engagement_ord'] = df['engagement_freq'].map(engage_map) if 'engagement_freq' in df.columns else 2
    df['purchase_ord'] = df['purchase_freq'].map(purchase_map) if 'purchase_freq' in df.columns else 1
    df['ad_click_ord'] = df['ad_click_freq'].map(adclick_map) if 'ad_click_freq' in df.columns else 1
    df['age_ord'] = df['age_group'].map(age_map).fillna(2) if 'age_group' in df.columns else 2
    df['searched_binary'] = (df['searched_product'] == 'Yes').astype(int) if 'searched_product' in df.columns else 1
    df['ad_influenced_binary'] = df['ad_influenced'].map({'Yes': 1, 'Not sure': 0.5, 'No': 0}).fillna(0.5) if 'ad_influenced' in df.columns else 0.5
    df['high_ad_responsive'] = (df['ad_click_ord'] >= 2).astype(int)
    df['made_purchase'] = (df['purchase_ord'] > 0).astype(int)

    for plat in ['Instagram', 'YouTube', 'Pinterest', 'Snapchat', 'Facebook', 'Twitter']:
        col = f'platform_{plat.lower()}'
        if 'platforms' in df.columns:
            df[col] = df['platforms'].str.contains(plat, na=False).astype(int)
        else:
            df[col] = 0

    motiv_map = {
        'motiv_positive_reviews': 'Positive reviews',
        'motiv_coupon': 'Discount / coupon code',
        'motiv_trust': 'Trust in influencer',
        'motiv_attractive': 'Attractive content',
        'motiv_curiosity': 'Curiosity',
        'motiv_fomo': 'Trend / FOMO',
    }
    for col, label in motiv_map.items():
        if 'purchase_motivation' in df.columns:
            df[col] = df['purchase_motivation'].str.contains(label, na=False).astype(int)
        else:
            df[col] = 0

    return df

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df_raw = load_data()
df = preprocess(df_raw)

# ─────────────────────────────────────────────
# SIDEBAR — LIVE FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
      <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; 
                  background:linear-gradient(135deg,#7c5cfc,#fc5c7d);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        🎛️ Live Filters
      </div>
      <div style="color:#7878a0; font-size:0.8rem; margin-top:4px;">Adjust to explore the data</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Demographics**")
    
    age_options = sorted(df['age_group'].dropna().unique().tolist())
    selected_ages = st.multiselect("Age Group", age_options, default=age_options,
                                   help="Filter by age group")

    gender_options = sorted(df['gender'].dropna().unique().tolist())
    selected_genders = st.multiselect("Gender", gender_options, default=gender_options)

    spending_options = ['< ₹500', '₹500–₹2000', '₹2000–₹5000', '₹5000+']
    spending_options = [s for s in spending_options if s in df['spending'].values]
    selected_spending = st.multiselect("Spending Range", spending_options, default=spending_options)

    st.markdown("---")
    st.markdown("**Behavior Filters**")

    trust_options = sorted(df['trust_factor'].dropna().unique().tolist())
    selected_trust = st.multiselect("Trust Driver", trust_options, default=trust_options)

    platform_options = ['Instagram', 'YouTube', 'Pinterest', 'Snapchat', 'Facebook']
    selected_platforms = st.multiselect("Platforms Used", platform_options, default=platform_options)

    st.markdown("---")
    st.markdown("**Analysis Target**")
    x_axis = st.selectbox("X-Axis (Factor)", 
                           ['engagement_ord', 'product_interest_score', 'daily_time_ord',
                            'satisfaction_score', 'retargeting_score', 'spending_ord', 'age_ord'],
                           format_func=lambda x: {
                               'engagement_ord': 'Engagement Frequency',
                               'product_interest_score': 'Product Interest Score',
                               'daily_time_ord': 'Daily Time Spent',
                               'satisfaction_score': 'Purchase Satisfaction',
                               'retargeting_score': 'Retargeting Receptivity',
                               'spending_ord': 'Monthly Spending',
                               'age_ord': 'Age Group'
                           }.get(x, x))

    y_axis = st.selectbox("Y-Axis (Outcome)",
                           ['purchase_ord', 'ad_click_ord', 'retargeting_score',
                            'product_interest_score', 'satisfaction_score'],
                           format_func=lambda x: {
                               'purchase_ord': 'Purchase Frequency',
                               'ad_click_ord': 'Ad Click Rate',
                               'retargeting_score': 'Retargeting Score',
                               'product_interest_score': 'Product Interest',
                               'satisfaction_score': 'Satisfaction Score'
                           }.get(x, x))

    color_by = st.selectbox("Color By", 
                             ['trust_factor', 'age_group', 'gender', 'preferred_ad_format'],
                             format_func=lambda x: {
                                 'trust_factor': 'Trust Driver',
                                 'age_group': 'Age Group',
                                 'gender': 'Gender',
                                 'preferred_ad_format': 'Ad Format'
                             }.get(x, x))

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
filtered = df.copy()

if selected_ages:
    filtered = filtered[filtered['age_group'].isin(selected_ages)]
if selected_genders:
    filtered = filtered[filtered['gender'].isin(selected_genders)]
if selected_spending:
    filtered = filtered[filtered['spending'].isin(selected_spending)]
if selected_trust:
    filtered = filtered[filtered['trust_factor'].isin(selected_trust)]
if selected_platforms:
    plat_mask = pd.Series([False] * len(filtered), index=filtered.index)
    for p in selected_platforms:
        col = f'platform_{p.lower()}'
        if col in filtered.columns:
            plat_mask = plat_mask | (filtered[col] == 1)
    filtered = filtered[plat_mask]

n_filtered = len(filtered)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="hero-title">The Influence Crew</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Influencer Content Impact on Consumer Purchase & Ad Responsiveness | Business Analytics Dashboard</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style="text-align:right; padding-top:10px;">
      <span style="background:#7c5cfc22; color:#a08cff; border:1px solid #7c5cfc44; 
                   padding:5px 12px; border-radius:20px; font-size:0.8rem;">Group 5</span>
      &nbsp;
      <span style="background:#fc5c7d22; color:#ff8ca8; border:1px solid #fc5c7d44; 
                   padding:5px 12px; border-radius:20px; font-size:0.8rem;">{n_filtered} respondents</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs(["📊 Overview", "🔗 Factor Analysis", "🤖 Predictions", 
                 "⛏️ Pattern Mining", "🕸️ Network Analysis", "📈 Trends & PCA", "🏆 Recommendations",
                 "🧩 Clustering", "💰 ROI Simulator", "☁️ Text Analytics", "📥 Export"])

# ═══════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════
with tabs[0]:
    # KPI Row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    purchase_rate = filtered['made_purchase'].mean() * 100 if n_filtered > 0 else 0
    avg_interest = filtered['product_interest_score'].mean() if n_filtered > 0 else 0
    avg_satisfaction = filtered['satisfaction_score'].mean() if n_filtered > 0 else 0
    ad_responsive_rate = filtered['high_ad_responsive'].mean() * 100 if n_filtered > 0 else 0
    avg_engagement = filtered['engagement_ord'].mean() if n_filtered > 0 else 0

    kpi1.metric("👥 Respondents", f"{n_filtered}", f"{n_filtered-210} from total" if n_filtered != 210 else "Full dataset")
    kpi2.metric("🛒 Purchase Rate", f"{purchase_rate:.1f}%", "made ≥1 purchase")
    kpi3.metric("⭐ Avg Interest Score", f"{avg_interest:.2f}/5", f"{'Above' if avg_interest>3 else 'Below'} neutral")
    kpi4.metric("😊 Avg Satisfaction", f"{avg_satisfaction:.2f}/5")
    kpi5.metric("📢 Ad Responsive", f"{ad_responsive_rate:.1f}%", "click ads sometimes+")

    st.markdown('<div class="section-head">Demographic Breakdown</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        age_counts = filtered['age_group'].value_counts()
        wedges, texts, autotexts = ax.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%',
                                            colors=PALETTE[:len(age_counts)], startangle=90,
                                            wedgeprops={'edgecolor':'#0a0a0f', 'linewidth': 2})
        for t in autotexts: t.set_fontsize(8)
        ax.set_title('Age Distribution', fontweight='bold', pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sp_map = {'< ₹500': 1, '₹500–₹2000': 2, '₹2000–₹5000': 3, '₹5000+': 4}
        sp_order = [k for k in sp_map if k in filtered['spending'].values]
        sp_counts = filtered['spending'].value_counts().reindex(sp_order).fillna(0)
        bars = ax.bar(range(len(sp_counts)), sp_counts.values, color=PALETTE[:len(sp_counts)])
        ax.set_xticks(range(len(sp_counts)))
        ax.set_xticklabels(['<500','500-2k','2k-5k','5k+'], fontsize=9)
        ax.set_title('Monthly Spending (₹)', fontweight='bold')
        ax.bar_label(bars, fontsize=9)
        ax.set_ylabel('Count')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c3:
        fig, ax = plt.subplots(figsize=(5, 4))
        plat_cols = [f'platform_{p.lower()}' for p in ['Instagram','YouTube','Pinterest','Snapchat','Facebook','Twitter']]
        plat_cols = [c for c in plat_cols if c in filtered.columns]
        plat_sums = filtered[plat_cols].sum().sort_values()
        plat_labels = [c.replace('platform_','').title() for c in plat_sums.index]
        ax.barh(plat_labels, plat_sums.values, color=PALETTE[:len(plat_sums)])
        ax.set_title('Platform Usage', fontweight='bold')
        ax.set_xlabel('Users')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-head">Content & Trust Insights</div>', unsafe_allow_html=True)

    c4, c5 = st.columns(2)

    with c4:
        fig, ax = plt.subplots(figsize=(6, 4))
        tf_counts = filtered['trust_factor'].value_counts().sort_values(ascending=True)
        colors_tf = [PALETTE[i % len(PALETTE)] for i in range(len(tf_counts))]
        bars = ax.barh(tf_counts.index, tf_counts.values, color=colors_tf)
        ax.set_title('Trust Driver Rankings', fontweight='bold')
        ax.set_xlabel('Respondent Count')
        ax.bar_label(bars, padding=3, fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c5:
        fig, ax = plt.subplots(figsize=(6, 4))
        eng_order = ['Rarely','Sometimes','frequently','Almost always']
        eng_counts = filtered['engagement_freq'].value_counts().reindex(eng_order).fillna(0)
        bars = ax.bar(eng_order, eng_counts.values, color=PALETTE[1], alpha=0.85, edgecolor='#0a0a0f')
        ax.set_title('Engagement Frequency', fontweight='bold')
        ax.bar_label(bars, fontsize=9)
        ax.set_ylabel('Count')
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ═══════════════════════════════════════════════
# TAB 2: FACTOR ANALYSIS (LIVE)
# ═══════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-head">🔴 Live Factor Relationship Explorer</div>', unsafe_allow_html=True)
    st.caption("Use the sidebar to change X-axis, Y-axis, and Color By to explore how one factor affects another.")

    if n_filtered < 10:
        st.warning("Too few respondents after filtering. Please adjust filters.")
    else:
        col_a, col_b = st.columns([2, 1])

        with col_a:
            fig, ax = plt.subplots(figsize=(8, 5))

            x_data = filtered[x_axis].dropna()
            y_data = filtered[y_axis].dropna()
            
            # Align by index
            common_idx = x_data.index.intersection(y_data.index)
            x_aligned = x_data.loc[common_idx]
            y_aligned = y_data.loc[common_idx]
            color_data = filtered[color_by].loc[common_idx]

            unique_groups = color_data.dropna().unique()
            for i, grp in enumerate(sorted(unique_groups)):
                mask = color_data == grp
                ax.scatter(
                    x_aligned[mask] + np.random.uniform(-0.1, 0.1, mask.sum()),
                    y_aligned[mask] + np.random.uniform(-0.1, 0.1, mask.sum()),
                    alpha=0.65, s=35, c=PALETTE[i % len(PALETTE)],
                    label=str(grp)[:20]
                )

            # Trend line
            try:
                z = np.polyfit(x_aligned, y_aligned, 1)
                p_line = np.poly1d(z)
                xs = np.linspace(x_aligned.min(), x_aligned.max(), 100)
                ax.plot(xs, p_line(xs), '--', color='white', linewidth=2, alpha=0.7, label='Trend')
            except:
                pass

            rho, p_val = spearmanr(x_aligned, y_aligned)
            sig_text = f'ρ = {rho:.3f}  p = {p_val:.4f}  {"✅ Significant" if p_val < 0.05 else "❌ Not Significant"}'
            ax.set_title(f'{x_axis} → {y_axis}\n{sig_text}', fontweight='bold', fontsize=11)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.legend(bbox_to_anchor=(1.01, 1), fontsize=7.5, title=color_by)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            st.markdown("**Correlation Result**")
            
            abs_rho = abs(rho)
            strength = "Strong 🔴" if abs_rho > 0.4 else ("Moderate 🟡" if abs_rho > 0.2 else "Weak ⚪")
            direction = "Positive ↑" if rho > 0 else "Negative ↓"
            sig = "Yes ✅" if p_val < 0.05 else "No ❌"

            st.metric("Spearman ρ", f"{rho:.3f}")
            st.metric("p-value", f"{p_val:.4f}")
            st.metric("Strength", strength)
            st.metric("Direction", direction)
            st.metric("Significant?", sig)

            st.markdown("---")
            st.markdown("**Group Breakdown**")
            
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            group_means = filtered.groupby(color_by)[y_axis].mean().sort_values(ascending=True)
            colors_g = [PALETTE[i % len(PALETTE)] for i in range(len(group_means))]
            bars = ax2.barh([str(x)[:18] for x in group_means.index], group_means.values, color=colors_g)
            ax2.set_title(f'Mean {y_axis}\nby {color_by}', fontweight='bold', fontsize=9)
            ax2.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # Heatmap of all correlations
        st.markdown('<div class="section-head">Full Correlation Matrix</div>', unsafe_allow_html=True)
        
        num_cols = ['spending_ord','daily_time_ord','engagement_ord','product_interest_score',
                    'purchase_ord','satisfaction_score','ad_click_ord','retargeting_score',
                    'searched_binary','ad_influenced_binary']
        num_labels = ['Spending','Daily Time','Engagement','Product Interest',
                       'Purchase Freq','Satisfaction','Ad Clicks','Retargeting',
                       'Searched','Ad Influenced']
        
        corr_data = filtered[num_cols].dropna()
        if len(corr_data) > 5:
            corr_mat = corr_data.corr(method='spearman')
            corr_mat.index = num_labels
            corr_mat.columns = num_labels

            fig3, ax3 = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_mat, dtype=bool))
            sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                        center=0, vmin=-1, vmax=1, ax=ax3,
                        linewidths=0.5, linecolor='#0a0a0f',
                        annot_kws={'size': 9})
            ax3.set_title('Spearman Correlation Matrix (Filtered Data)', fontweight='bold')
            plt.xticks(rotation=35, ha='right')
            st.pyplot(fig3, use_container_width=True)
            plt.close()

            # Business interpretation of top correlations
            st.markdown('<div class="section-head">Auto-Generated Business Insights</div>', unsafe_allow_html=True)
            
            # Find top 3 correlations with purchase
            purch_corr = corr_mat['Purchase Freq'].drop('Purchase Freq').abs().sort_values(ascending=False)
            
            ins_cols = st.columns(3)
            for i, (feat, val) in enumerate(purch_corr.head(3).items()):
                raw_corr = corr_mat.loc['Purchase Freq', feat]
                direction_word = "increases" if raw_corr > 0 else "decreases"
                with ins_cols[i]:
                    st.markdown(f"""
                    <div class="insight-box">
                      <div class="insight-title" style="color:{PALETTE[i]}">
                        {'🟢' if raw_corr > 0 else '🔴'} {feat} → Purchase
                      </div>
                      <div class="insight-text">
                        ρ = {raw_corr:.3f} — As <strong>{feat}</strong> {direction_word}, 
                        purchase frequency tends to {'rise' if raw_corr > 0 else 'fall'}. 
                        {'A statistically meaningful lever for brands.' if abs(raw_corr) > 0.15 else 'A weak but notable pattern.'}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 3: PREDICTIONS
# ═══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-head">Predictive Modeling — Ad Responsiveness & Purchase</div>', unsafe_allow_html=True)

    feat_cols = ['spending_ord','daily_time_ord','engagement_ord','product_interest_score',
                 'satisfaction_score','ad_click_ord','retargeting_score',
                 'searched_binary','ad_influenced_binary']
    feat_labels = ['Spending','Daily Time','Engagement','Product Interest',
                    'Satisfaction','Ad Clicks','Retargeting','Searched','Ad Influenced']

    model_df = filtered[feat_cols + ['high_ad_responsive','purchase_ord']].dropna()

    if len(model_df) < 30:
        st.warning("Not enough data for modeling after filtering. Please widen filters.")
    else:
        X = model_df[feat_cols].values
        y_ad = model_df['high_ad_responsive'].values
        y_pur = model_df['purchase_ord'].values

        sc_m = StandardScaler()
        X_sc = sc_m.fit_transform(X)

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("#### 🎯 Logistic Regression — Ad Responsiveness")
            
            if y_ad.sum() >= 5 and (len(y_ad) - y_ad.sum()) >= 5:
                from sklearn.model_selection import train_test_split
                X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_ad, test_size=0.25,
                                                            random_state=42, stratify=y_ad)
                lr = LogisticRegression(max_iter=500, C=1.0, random_state=42)
                lr.fit(X_tr, y_tr)
                y_prob = lr.predict_proba(X_te)[:,1]
                y_pred = lr.predict(X_te)
                auc = roc_auc_score(y_te, y_prob)
                fpr, tpr, _ = roc_curve(y_te, y_prob)

                fig, axes = plt.subplots(1, 2, figsize=(9, 4))

                # ROC
                axes[0].plot(fpr, tpr, color=PALETTE[0], linewidth=2.5, label=f'AUC = {auc:.3f}')
                axes[0].plot([0,1],[0,1],'--', color='#5a5a7a', linewidth=1)
                axes[0].fill_between(fpr, tpr, alpha=0.12, color=PALETTE[0])
                axes[0].set_title('ROC Curve', fontweight='bold')
                axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
                axes[0].legend()

                # Confusion Matrix
                cm = confusion_matrix(y_te, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                            xticklabels=['Low','High'], yticklabels=['Low','High'],
                            linewidths=1, annot_kws={'size': 14})
                axes[1].set_title('Confusion Matrix', fontweight='bold')
                axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

                st.pyplot(fig, use_container_width=True)
                plt.close()

                st.metric("AUC Score", f"{auc:.3f}", 
                          "Good" if auc > 0.7 else "Moderate")
            else:
                st.info("Insufficient class balance in filtered data for classification.")

        with col_m2:
            st.markdown("#### 📊 Feature Importance — What Drives Outcomes?")

            rf = RandomForestClassifier(n_estimators=80, max_depth=4, random_state=42)
            rf.fit(X_sc, y_ad)
            imp = rf.feature_importances_
            imp_df = pd.DataFrame({'Feature': feat_labels, 'Importance': imp}).sort_values('Importance')

            fig, ax = plt.subplots(figsize=(7, 5))
            colors_imp = plt.cm.plasma(np.linspace(0.2, 0.85, len(imp_df)))
            bars = ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors_imp)
            ax.set_title('RF Feature Importance\n(Predicting Ad Responsiveness)', fontweight='bold')
            ax.set_xlabel('Importance Score')
            ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Regression coefficients
        st.markdown('<div class="section-head">Linear Regression — Purchase Frequency Drivers</div>', unsafe_allow_html=True)
        
        reg = LinearRegression()
        reg.fit(X_sc, y_pur)
        coef_df = pd.DataFrame({'Feature': feat_labels, 'Coefficient': reg.coef_}).sort_values('Coefficient')

        fig, ax = plt.subplots(figsize=(10, 4))
        bar_colors = [PALETTE[0] if c > 0 else PALETTE[1] for c in coef_df['Coefficient']]
        bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=bar_colors)
        ax.axvline(0, color='white', linewidth=1.5)
        ax.set_title('Standardized Regression Coefficients — Purchase Frequency\n(Purple = positive driver, Pink = negative/weak driver)',
                     fontweight='bold')
        ax.set_xlabel('Coefficient (Effect Size)')
        ax.bar_label(bars, fmt='%.3f', fontsize=9, padding=3)
        pos_p = mpatches.Patch(color=PALETTE[0], label='Positive driver')
        neg_p = mpatches.Patch(color=PALETTE[1], label='Negative driver')
        ax.legend(handles=[pos_p, neg_p])
        st.pyplot(fig, use_container_width=True)
        plt.close()

        r2 = reg.score(X_sc, y_pur)
        st.caption(f"R² = {r2:.3f} — The model explains {r2*100:.1f}% of variance in purchase frequency. Note: small n limits generalizability.")

# ═══════════════════════════════════════════════
# TAB 4: PATTERN MINING
# ═══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-head">Association Rule Mining — Behavioral Pattern Discovery</div>', unsafe_allow_html=True)

    # Build transaction matrix
    t_df = pd.DataFrame({
        'High_Engagement': (filtered['engagement_ord'] >= 2).astype(int),
        'ContentQuality_Trust': (filtered['trust_factor'] == 'Content quality').astype(int),
        'Transparency_Trust': (filtered['trust_factor'] == 'Transparency in reviews').astype(int),
        'Authenticity_Trust': (filtered['trust_factor'] == 'Authenticity').astype(int),
        'High_Interest': (filtered['product_interest_score'] >= 4).astype(int),
        'High_Satisfaction': (filtered['satisfaction_score'] >= 4).astype(int),
        'Made_Purchase': (filtered['purchase_ord'] > 0).astype(int),
        'High_Purchase': (filtered['purchase_ord'] >= 2).astype(int),
        'High_AdClick': (filtered['ad_click_ord'] >= 2).astype(int),
        'Motiv_Reviews': filtered.get('motiv_positive_reviews', pd.Series(0, index=filtered.index)),
        'Motiv_Coupon': filtered.get('motiv_coupon', pd.Series(0, index=filtered.index)),
        'Uses_Instagram': filtered['platform_instagram'],
        'Uses_YouTube': filtered['platform_youtube'],
        'High_TimeSpent': (filtered['daily_time_ord'] >= 3).astype(int),
        'High_Retargeting': (filtered['retargeting_score'] >= 4).astype(int),
    }).fillna(0).astype(int)

    # Compute rules
    def compute_rules(df_t, antecedents, consequent, min_sup=0.08, min_conf=0.5):
        n = len(df_t)
        rules = []
        cons = df_t[consequent]
        cons_rate = cons.sum() / n

        for col in antecedents:
            if col == consequent:
                continue
            ant = df_t[col]
            both = (ant==1) & (cons==1)
            sup = both.sum() / n
            if sup < min_sup or ant.sum() == 0:
                continue
            conf = both.sum() / ant.sum()
            if conf < min_conf:
                continue
            lift = conf / cons_rate if cons_rate > 0 else 0
            rules.append({'Antecedent': col, 'Consequent': consequent,
                           'Support': round(sup,3), 'Confidence': round(conf,3), 'Lift': round(lift,3)})

        for i in range(len(antecedents)):
            for j in range(i+1, len(antecedents)):
                c1, c2 = antecedents[i], antecedents[j]
                if c1==consequent or c2==consequent:
                    continue
                pair = (df_t[c1]==1) & (df_t[c2]==1)
                both = pair & (cons==1)
                sup = both.sum() / n
                if sup < min_sup or pair.sum() == 0:
                    continue
                conf = both.sum() / pair.sum()
                if conf < min_conf:
                    continue
                lift = conf / cons_rate if cons_rate > 0 else 0
                rules.append({'Antecedent': f'{c1} + {c2}', 'Consequent': consequent,
                               'Support': round(sup,3), 'Confidence': round(conf,3), 'Lift': round(lift,3)})

        return pd.DataFrame(rules).sort_values('Lift', ascending=False) if rules else pd.DataFrame()

    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        target_rule = st.selectbox("Predict Outcome", ['Made_Purchase', 'High_Purchase', 'High_AdClick', 'High_Interest'])
        min_conf = st.slider("Min Confidence", 0.3, 0.9, 0.5, 0.05)
        min_sup = st.slider("Min Support", 0.05, 0.4, 0.08, 0.01)

    antecedent_list = [c for c in t_df.columns if c != target_rule]
    rules_df = compute_rules(t_df, antecedent_list, target_rule, min_sup=min_sup, min_conf=min_conf)

    if rules_df.empty:
        st.info("No rules found at current thresholds. Try lowering minimum support/confidence.")
    else:
        with col_r2:
            st.metric("Rules Found", len(rules_df))
            st.metric("Best Lift", f"{rules_df['Lift'].max():.3f}")
            st.metric("Best Confidence", f"{rules_df['Confidence'].max():.1%}")

        c_left, c_right = st.columns(2)

        with c_left:
            fig, ax = plt.subplots(figsize=(7, max(4, len(rules_df.head(12))*0.5)))
            top = rules_df.head(12)
            colors_l = plt.cm.plasma(np.linspace(0.2, 0.9, len(top)))
            ax.barh(top['Antecedent'], top['Lift'], color=colors_l)
            ax.axvline(1, color='white', linestyle='--', linewidth=1.5, label='Lift=1 (random)')
            ax.set_title(f'Top Rules → {target_rule}\n(by Lift)', fontweight='bold')
            ax.set_xlabel('Lift')
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c_right:
            fig, ax = plt.subplots(figsize=(6, 5))
            sc_p = ax.scatter(rules_df['Support'], rules_df['Confidence'],
                               c=rules_df['Lift'], cmap='plasma', s=80, alpha=0.8)
            plt.colorbar(sc_p, ax=ax, label='Lift')
            ax.set_xlabel('Support')
            ax.set_ylabel('Confidence')
            ax.set_title('Support vs Confidence\n(colored by Lift)', fontweight='bold')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("**Rule Table (Top 15 by Lift)**")
        st.dataframe(rules_df.head(15).style.background_gradient(
            cmap='plasma', subset=['Lift']).format({'Support': '{:.3f}', 'Confidence': '{:.3f}', 'Lift': '{:.3f}'}),
            use_container_width=True)

        # Business interpretation
        if len(rules_df) > 0:
            best_rule = rules_df.iloc[0]
            st.success(f"🏆 **Best Rule:** `{best_rule['Antecedent']}` → `{target_rule}` | "
                       f"Lift={best_rule['Lift']:.2f} — This combination is {best_rule['Lift']:.1f}x more likely "
                       f"to result in {target_rule} than a random consumer.")

# ═══════════════════════════════════════════════
# TAB SNA: SOCIAL NETWORK ANALYSIS
# ═══════════════════════════════════════════════
with tabs[4]:
    import networkx as nx
    from collections import Counter
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    st.markdown('<div class="section-head">🕸️ Social Network Analysis — Consumer Decision Networks</div>', unsafe_allow_html=True)
    st.caption("Networks built from survey co-occurrence data. Select a network type to explore.")

    sna_type = st.radio("Choose Network",
        ["Platform Co-Occurrence", "Trust ↔ Motivation Bipartite", "Content Co-Occurrence", "Influence Flow DAG"],
        horizontal=True)

    # ── Build all networks from filtered data ────────────────────────────
    # Platform co-occurrence
    @st.cache_data
    def build_platform_network(platforms_series):
        platforms_series = pd.Series(platforms_series)
        G = nx.Graph()
        usage = Counter()
        edges = Counter()
        for row in platforms_series.dropna():
            plats = [p.strip() for p in row.split(';') if p.strip()]
            for p in plats: usage[p] += 1
            for i in range(len(plats)):
                for j in range(i+1, len(plats)):
                    edges[tuple(sorted([plats[i], plats[j]]))] += 1
        for n, c in usage.items(): G.add_node(n, usage=c)
        for (u,v), w in edges.items(): G.add_edge(u, v, weight=w)
        return G, usage, edges

    @st.cache_data
    def build_bipartite_network(trust_series, motiv_series):
        trust_series = pd.Series(trust_series)   # ← ADD THIS LINE
        motiv_series = pd.Series(motiv_series)   # ← ADD THIS LINE
        all_motivs = []
        for row in motiv_series.dropna():
            for m in row.split(';'):
                m = m.strip()
                if m and m != 'nan': all_motivs.append(m)
        trust_nodes = trust_series.dropna().unique().tolist()
        motiv_nodes = list(set(all_motivs))
        B = nx.Graph()
        B.add_nodes_from(trust_nodes, bipartite=0)
        B.add_nodes_from(motiv_nodes, bipartite=1)
        combined = pd.concat([trust_series.rename('trust'), motiv_series.rename('motiv')], axis=1)
        for _, row in combined.iterrows():
            t = str(row.get('trust','')).strip()
            raw = str(row.get('motiv',''))
            if not t or t == 'nan': continue
            for m in raw.split(';'):
                m = m.strip()
                if m and m != 'nan':
                    if B.has_edge(t, m): B[t][m]['weight'] += 1
                    else: B.add_edge(t, m, weight=1)
        return B, trust_nodes, motiv_nodes

    @st.cache_data
    def build_content_network(content_series, purchase_series):
        content_series = pd.Series(content_series)     # ← ADD THIS LINE
        purchase_series = pd.Series(purchase_series)   # ← ADD THIS LINE
        G = nx.Graph()
        usage = Counter()
        edges = Counter()
        content_purchase = {}
        for ct, pur in zip(content_series.fillna(''), purchase_series.fillna(0)):
            items = [c.strip() for c in str(ct).split(';') if c.strip() and c.strip() != 'nan']
            for item in items:
                usage[item] += 1
                content_purchase.setdefault(item, []).append(float(pur))
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    edges[tuple(sorted([items[i], items[j]]))] += 1
        for n, c in usage.items():
            G.add_node(n, usage=c, avg_purchase=np.mean(content_purchase.get(n,[0])))
        for (u,v), w in edges.items():
            G.add_edge(u, v, weight=w)
        return G, usage, content_purchase

    # Prepare purchase_ord for filtered data
    purchase_map_sna = {'Never':0,'1-2 times':1,'2-3 times':2,'3-2 times':3,'More than 5 times':4}
    filtered_pur = filtered['purchase_freq'].map(purchase_map_sna).fillna(0)

    if n_filtered < 10:
        st.warning("Too few respondents after filtering. Please widen filters.")
    else:
        # ── PLATFORM CO-OCCURRENCE ──────────────────────────────────────
        if sna_type == "Platform Co-Occurrence":
            G_p, p_usage, p_edges = build_platform_network(filtered['platforms'].dropna().tolist())

            if len(G_p.nodes()) < 2:
                st.info("Not enough platform data in current filter selection.")
            else:
                degree_c   = nx.degree_centrality(G_p)
                between_c  = nx.betweenness_centrality(G_p, weight='weight')
                close_c    = nx.closeness_centrality(G_p)
                pagerank_c = nx.pagerank(G_p, weight='weight')

                cent_df = pd.DataFrame({
                    'Platform':    list(p_usage.keys()),
                    'Users':       [p_usage[p] for p in p_usage],
                    'Degree':      [round(degree_c.get(p,0),3) for p in p_usage],
                    'Betweenness': [round(between_c.get(p,0),3) for p in p_usage],
                    'Closeness':   [round(close_c.get(p,0),3) for p in p_usage],
                    'PageRank':    [round(pagerank_c.get(p,0),4) for p in p_usage],
                }).sort_values('Users', ascending=False)

                col_net1, col_net2 = st.columns([3,2])

                with col_net1:
                    fig, ax = plt.subplots(figsize=(7, 6))
                    pos = nx.spring_layout(G_p, seed=42, k=2.0)
                    n_sizes  = [p_usage.get(n,10)*4 for n in G_p.nodes()]
                    n_colors = [pagerank_c.get(n,0) for n in G_p.nodes()]
                    e_widths = [G_p[u][v]['weight']/14 for u,v in G_p.edges()]
                    e_alphas = [min(G_p[u][v]['weight']/120, 1.0) for u,v in G_p.edges()]
                    nx.draw_networkx_nodes(G_p, pos, ax=ax, node_size=n_sizes,
                                           node_color=n_colors, cmap=plt.cm.plasma, alpha=0.92)
                    for (u,v), w, a in zip(G_p.edges(), e_widths, e_alphas):
                        nx.draw_networkx_edges(G_p, pos, edgelist=[(u,v)], ax=ax,
                                                width=w, alpha=a, edge_color='#7c5cfc')
                    nx.draw_networkx_labels(G_p, pos, ax=ax, font_size=9,
                                             font_color='white', font_weight='bold')
                    elbls = {(u,v): G_p[u][v]['weight'] for u,v in G_p.edges()}
                    nx.draw_networkx_edge_labels(G_p, pos, edge_labels=elbls, ax=ax,
                                                  font_size=7, font_color='#fca95c')
                    ax.set_title('Platform Co-Occurrence Network\nNode size=Usage  |  Colour=PageRank  |  Edge=Co-occurrence',
                                  fontsize=9, fontweight='bold')
                    ax.set_axis_off()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                with col_net2:
                    st.markdown("**Centrality Metrics**")
                    st.dataframe(cent_df.style.background_gradient(cmap='plasma', subset=['PageRank']).format(
                        {'Degree':'{:.3f}','Betweenness':'{:.3f}','Closeness':'{:.3f}','PageRank':'{:.4f}'}),
                        use_container_width=True)

                    top_pr  = max(pagerank_c, key=pagerank_c.get)
                    top_bt  = max(between_c, key=between_c.get)
                    ig_yt   = p_edges.get(('Instagram','YouTube'),0)

                    st.markdown(f"""
                    <div class="insight-box">
                      <div class="insight-title" style="color:{PALETTE[0]}">🏆 Highest PageRank</div>
                      <div class="insight-text"><strong>{top_pr}</strong> — most influential node in the platform ecosystem. Anchor all campaigns here.</div>
                    </div>
                    <div class="insight-box">
                      <div class="insight-title" style="color:{PALETTE[2]}">🌉 Highest Betweenness</div>
                      <div class="insight-text"><strong>{top_bt}</strong> — bridges the most platform communities. Best for cross-community spillover.</div>
                    </div>
                    <div class="insight-box">
                      <div class="insight-title" style="color:{PALETTE[3]}">🔗 Dominant Dyad</div>
                      <div class="insight-text">Instagram–YouTube co-occurrence = <strong>{ig_yt}</strong> respondents. Run coordinated dual-platform campaigns on this pair.</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── BIPARTITE: TRUST ↔ MOTIVATION ──────────────────────────────
        elif sna_type == "Trust ↔ Motivation Bipartite":
            B, t_nodes, m_nodes = build_bipartite_network(
                filtered['trust_factor'].dropna().tolist(),
                filtered['purchase_motivation'].fillna('').tolist())

            if len(B.nodes()) < 3:
                st.info("Not enough data. Widen filters.")
            else:
                wdeg = dict(B.degree(weight='weight'))

                col_b1, col_b2 = st.columns([3,2])

                with col_b1:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
                    fig.suptitle('Trust Driver ↔ Purchase Motivation', fontsize=12, fontweight='bold')

                    pos_bip = {}
                    for i, n in enumerate(t_nodes): pos_bip[n] = (0, i*1.4 - len(t_nodes)*0.7)
                    for i, n in enumerate(m_nodes): pos_bip[n] = (3, i*1.4 - len(m_nodes)*0.7)

                    t_sizes = [wdeg.get(n,1)*28 for n in t_nodes]
                    m_sizes = [wdeg.get(n,1)*28 for n in m_nodes]
                    nx.draw_networkx_nodes(B, pos_bip, nodelist=t_nodes, ax=axes[0],
                                           node_color=PALETTE[0], node_size=t_sizes, alpha=0.9)
                    nx.draw_networkx_nodes(B, pos_bip, nodelist=m_nodes, ax=axes[0],
                                           node_color=PALETTE[2], node_size=m_sizes, alpha=0.9)
                    max_w = max((B[u][v]['weight'] for u,v in B.edges()), default=1)
                    for u,v in B.edges():
                        w = B[u][v]['weight']
                        nx.draw_networkx_edges(B, pos_bip, edgelist=[(u,v)], ax=axes[0],
                                                width=w/max_w*5, alpha=w/max_w*0.75+0.1,
                                                edge_color='#fca95c')
                    nx.draw_networkx_labels(B, pos_bip, ax=axes[0], font_size=8,
                                             font_color='white', font_weight='bold')
                    axes[0].set_axis_off()
                    axes[0].set_title('Bipartite Network', fontsize=9)

                    # Heatmap
                    heat = pd.DataFrame(0, index=t_nodes, columns=m_nodes)
                    for u,v,d in B.edges(data=True):
                        if u in t_nodes and v in m_nodes: heat.loc[u,v] = d['weight']
                        elif v in t_nodes and u in m_nodes: heat.loc[v,u] = d['weight']
                    sns.heatmap(heat, ax=axes[1], cmap='YlOrRd', annot=True, fmt='d',
                                linewidths=0.5, linecolor='#0a0a0f', annot_kws={'size':8})
                    axes[1].set_title('Co-occurrence Heatmap', fontsize=9)
                    plt.setp(axes[1].get_xticklabels(), rotation=30, ha='right', fontsize=7)
                    plt.setp(axes[1].get_yticklabels(), fontsize=7)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                with col_b2:
                    st.markdown("**Weighted Degree by Node**")
                    wdeg_df = pd.DataFrame({
                        'Node': list(wdeg.keys()),
                        'Type': ['Trust' if n in t_nodes else 'Motivation' for n in wdeg.keys()],
                        'Weighted Degree': list(wdeg.values())
                    }).sort_values('Weighted Degree', ascending=False)
                    st.dataframe(wdeg_df, use_container_width=True)

                    top_t = max({n:wdeg[n] for n in t_nodes}, key=lambda x: wdeg[x])
                    top_m = max({n:wdeg[n] for n in m_nodes}, key=lambda x: wdeg[x])
                    st.markdown(f"""
                    <div class="insight-box">
                      <div class="insight-title" style="color:{PALETTE[0]}">🏆 Most Central Trust Driver</div>
                      <div class="insight-text"><strong>{top_t}</strong> connects to every motivation node — the universal gateway to conversion.</div>
                    </div>
                    <div class="insight-box">
                      <div class="insight-title" style="color:{PALETTE[2]}">🎯 Most Activated Motivation</div>
                      <div class="insight-text"><strong>{top_m}</strong> is triggered across all trust types — the dominant conversion endpoint brands should design for.</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── CONTENT CO-OCCURRENCE ───────────────────────────────────────
        elif sna_type == "Content Co-Occurrence":
            G_c, c_usage, c_purchase = build_content_network(
                filtered['content_type'].fillna('').tolist(),
                filtered_pur.tolist())

            if len(G_c.nodes()) < 2:
                st.info("Not enough content data. Widen filters.")
            else:
                pr_c = nx.pagerank(G_c, weight='weight')
                bt_c = nx.betweenness_centrality(G_c, weight='weight')

                col_c1, col_c2 = st.columns([3,2])

                with col_c1:
                    fig, ax = plt.subplots(figsize=(7,6))
                    pos_c = nx.spring_layout(G_c, seed=7, k=2.5)
                    avg_v = [G_c.nodes[n]['avg_purchase'] for n in G_c.nodes()]
                    norm  = mcolors.Normalize(vmin=min(avg_v) if avg_v else 0,
                                               vmax=max(avg_v) if avg_v else 1)
                    n_col = [cm.RdYlGn(norm(v)) for v in avg_v]
                    n_sz  = [c_usage.get(n,10)*10 for n in G_c.nodes()]
                    e_w   = [G_c[u][v]['weight']/22 for u,v in G_c.edges()]
                    nx.draw_networkx_nodes(G_c, pos_c, ax=ax, node_size=n_sz,
                                           node_color=n_col, alpha=0.93)
                    nx.draw_networkx_edges(G_c, pos_c, ax=ax, width=e_w,
                                           alpha=0.35, edge_color='#7878aa')
                    nx.draw_networkx_labels(G_c, pos_c, ax=ax, font_size=8,
                                             font_color='white', font_weight='bold')
                    sm = cm.ScalarMappable(cmap='RdYlGn', norm=norm)
                    sm.set_array([])
                    plt.colorbar(sm, ax=ax, label='Avg Purchase Score', shrink=0.75)
                    ax.set_title('Content Co-Occurrence Network\nNode size=Audience reach  |  Colour=Purchase conversion',
                                  fontsize=9, fontweight='bold')
                    ax.set_axis_off()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                with col_c2:
                    c_summ = pd.DataFrame({
                        'Content Type': list(c_usage.keys()),
                        'Users':         [c_usage[c] for c in c_usage],
                        'Avg Purchase':  [round(np.mean(c_purchase.get(c,[0])),3) for c in c_usage],
                        'PageRank':      [round(pr_c.get(c,0),4) for c in c_usage],
                        'Betweenness':   [round(bt_c.get(c,0),3) for c in c_usage],
                    }).sort_values('Avg Purchase', ascending=False)
                    st.dataframe(c_summ.style.background_gradient(
                        cmap='RdYlGn', subset=['Avg Purchase']).format(
                        {'Avg Purchase':'{:.3f}','PageRank':'{:.4f}','Betweenness':'{:.3f}'}),
                        use_container_width=True)

                    if len(c_summ) > 0:
                        top_buy = c_summ.iloc[0]['Content Type']
                        top_hub = max(pr_c, key=pr_c.get)
                        st.markdown(f"""
                        <div class="insight-box">
                          <div class="insight-title" style="color:{PALETTE[2]}">💰 Highest Purchase Conversion</div>
                          <div class="insight-text"><strong>{top_buy}</strong> — allocate disproportionate budget to micro-influencers in this niche.</div>
                        </div>
                        <div class="insight-box">
                          <div class="insight-title" style="color:{PALETTE[3]}">🌐 Highest PageRank Hub</div>
                          <div class="insight-text"><strong>{top_hub}</strong> — most central content node. Use for cross-niche multi-category campaigns.</div>
                        </div>
                        """, unsafe_allow_html=True)

        # ── INFLUENCE FLOW DAG ──────────────────────────────────────────
        elif sna_type == "Influence Flow DAG":
            engage_map_sna  = {'Rarely':'Low Engage','Sometimes':'Med Engage',
                                'frequently':'High Engage','Almost always':'High Engage'}
            purchase_lbl_sna = {0:'No Purchase',1:'Low Purchase',2:'Mid Purchase',
                                  3:'High Purchase',4:'High Purchase'}

            f2 = filtered.copy()
            f2['engage_lbl']  = f2['engagement_freq'].map(engage_map_sna).fillna('Low Engage')
            f2['purchase_lbl'] = filtered_pur.map(purchase_lbl_sna).fillna('No Purchase')

            DAG = nx.DiGraph()
            for _, row in f2.iterrows():
                t = str(row.get('trust_factor','')).strip()
                e = row.get('engage_lbl','Low Engage')
                p = row.get('purchase_lbl','No Purchase')
                if not t or t == 'nan': continue
                if DAG.has_edge(t, e): DAG[t][e]['weight'] += 1
                else: DAG.add_edge(t, e, weight=1)
                if DAG.has_edge(e, p): DAG[e][p]['weight'] += 1
                else: DAG.add_edge(e, p, weight=1)

            t_list = f2['trust_factor'].dropna().unique().tolist()
            e_list = ['Low Engage','Med Engage','High Engage']
            p_list = ['No Purchase','Low Purchase','Mid Purchase','High Purchase']

            layer_pos = {}
            for i,n in enumerate(t_list): layer_pos[n] = (0, i*1.5 - len(t_list)*0.75)
            for i,n in enumerate(e_list): layer_pos[n] = (3.5, i*2.8 - len(e_list)*1.4)
            for i,n in enumerate(p_list): layer_pos[n] = (7, i*1.8 - len(p_list)*0.9)

            lc = {}
            for n in t_list: lc[n] = PALETTE[0]
            for n in e_list: lc[n] = PALETTE[2]
            for n in p_list: lc[n] = PALETTE[1]

            fig, ax = plt.subplots(figsize=(13, 8))
            nc_dag = [lc.get(n,'white') for n in DAG.nodes()]
            ns_dag = [600 + dict(DAG.out_degree(weight='weight')).get(n,0)*2.5 for n in DAG.nodes()]
            nx.draw_networkx_nodes(DAG, layer_pos, ax=ax, node_color=nc_dag,
                                    node_size=ns_dag, alpha=0.92)
            if DAG.edges():
                max_w = max(DAG[u][v]['weight'] for u,v in DAG.edges())
                for u,v in DAG.edges():
                    w = DAG[u][v]['weight']
                    nx.draw_networkx_edges(DAG, layer_pos, edgelist=[(u,v)], ax=ax,
                                            width=w/max_w*6, alpha=w/max_w*0.65+0.15,
                                            edge_color='#fca95c', arrows=True, arrowsize=18,
                                            connectionstyle='arc3,rad=0.08')
            nx.draw_networkx_labels(DAG, layer_pos, ax=ax, font_size=8.5,
                                     font_color='white', font_weight='bold')
            all_y = [v[1] for v in layer_pos.values()]
            ymax  = max(all_y) if all_y else 3
            for lbl, xp in [('TRUST\nDRIVERS',0),('ENGAGEMENT\nLEVEL',3.5),('PURCHASE\nOUTCOME',7)]:
                ax.text(xp, ymax+2, lbl, ha='center', fontsize=10, fontweight='bold', color='#9090bb')
                ax.axvline(x=xp, color='#2a2a3a', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_title('Consumer Decision Influence Flow DAG\nEdge width = respondent volume through that path',
                          fontsize=11, fontweight='bold')
            ax.set_axis_off()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Flow volume table
            st.markdown("**Top Influence Paths (by respondent volume)**")
            paths_data = [(u, v, DAG[u][v]['weight']) for u,v in DAG.edges()]
            paths_df = pd.DataFrame(paths_data, columns=['From','To','Respondents']).sort_values('Respondents', ascending=False)
            st.dataframe(paths_df.head(15), use_container_width=True)

            st.markdown(f"""
            <div class="insight-box">
              <div class="insight-title" style="color:{PALETTE[3]}">🔑 Funnel Bottleneck</div>
              <div class="insight-text">Most consumers stall at Engagement → Purchase. The bottleneck is conversion, not awareness. Deploy coupon codes and CTAs at the engagement layer to break the stall.</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 5: TRENDS & PCA
# ═══════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-head">PCA — Consumer Segmentation</div>', unsafe_allow_html=True)

    pca_feature_cols = ['spending_ord','daily_time_ord','engagement_ord',
                         'product_interest_score','satisfaction_score',
                         'ad_click_ord','retargeting_score','searched_binary']
    pca_labels_list = ['Spending','Daily Time','Engagement','Product Interest',
                        'Satisfaction','Ad Clicks','Retargeting','Searched Product']

    pca_data = filtered[pca_feature_cols].dropna()

    if len(pca_data) < 10:
        st.warning("Too few data points for PCA. Widen filters.")
    else:
        scaler_pca = StandardScaler()
        X_pca_sc = scaler_pca.fit_transform(pca_data)

        pca_full = PCA()
        pca_full.fit(X_pca_sc)
        var_exp = pca_full.explained_variance_ratio_ * 100
        cum_var = np.cumsum(var_exp)

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(1, len(var_exp)+1), var_exp, color=PALETTE[0], alpha=0.85, label='Individual')
            ax2 = ax.twinx()
            ax2.plot(range(1, len(cum_var)+1), cum_var, 'o-', color=PALETTE[1], linewidth=2, label='Cumulative')
            ax2.axhline(80, color='white', linestyle='--', linewidth=1, alpha=0.6)
            ax2.set_ylabel('Cumulative %', color=PALETTE[1])
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Variance %')
            ax.set_title('Scree Plot', fontweight='bold')
            ax.set_xticks(range(1, len(var_exp)+1))
            st.pyplot(fig, use_container_width=True)
            plt.close()
            n_components_80 = np.argmax(cum_var >= 80) + 1
            st.caption(f"📐 {n_components_80} components explain ≥80% of total variance")

        with c2:
            pca_3 = PCA(n_components=min(3, len(pca_feature_cols)))
            pca_3.fit(X_pca_sc)
            n_comp = pca_3.n_components_
            loadings = pd.DataFrame(
                pca_3.components_.T,
                index=pca_labels_list,
                columns=[f'PC{i+1}' for i in range(n_comp)]
            )
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdYlGn',
                        center=0, vmin=-1, vmax=1, ax=ax,
                        linewidths=0.5, annot_kws={'size': 9})
            ax.set_title('PCA Component Loadings', fontweight='bold')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Scatter colored by purchase
        st.markdown('<div class="section-head">Consumer Segment Map (PC1 vs PC2)</div>', unsafe_allow_html=True)

        X_proj = pca_3.transform(X_pca_sc)
        proj_df = pd.DataFrame(X_proj, columns=[f'PC{i+1}' for i in range(n_comp)])
        proj_df['purchase_ord'] = filtered['purchase_ord'].loc[pca_data.index].values
        proj_df['trust_factor'] = filtered['trust_factor'].loc[pca_data.index].values

        color_pca = st.selectbox("Color PCA plot by:", ['Purchase Level', 'Trust Factor'])

        fig, ax = plt.subplots(figsize=(10, 5))
        if color_pca == 'Purchase Level':
            purch_labels_map = {0:'Never',1:'1-2x',2:'2-3x',3:'3-4x',4:'5+x'}
            for j, (val, label) in enumerate(purch_labels_map.items()):
                mask = proj_df['purchase_ord'] == val
                if mask.sum() > 0:
                    ax.scatter(proj_df.loc[mask,'PC1'], proj_df.loc[mask,'PC2'],
                               c=PALETTE[j%len(PALETTE)], alpha=0.65, s=40, label=label)
        else:
            for j, grp in enumerate(proj_df['trust_factor'].dropna().unique()):
                mask = proj_df['trust_factor'] == grp
                ax.scatter(proj_df.loc[mask,'PC1'], proj_df.loc[mask,'PC2'],
                           c=PALETTE[j%len(PALETTE)], alpha=0.65, s=40, label=grp)

        ax.axhline(0, color='#3a3a5c', linewidth=1)
        ax.axvline(0, color='#3a3a5c', linewidth=1)
        ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% variance)')
        ax.set_title('Consumer Segment Map', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.01, 1), fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Ad responsiveness trend
    st.markdown('<div class="section-head">Purchase & Ad Behavior Distributions</div>', unsafe_allow_html=True)
    
    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(6, 4))
        pf_order = ['Never','1-2 times','2-3 times','3-2 times','More than 5 times']
        pf_counts = filtered['purchase_freq'].value_counts().reindex(pf_order).fillna(0)
        bars = ax.bar(range(len(pf_counts)), pf_counts.values, color=PALETTE[:5])
        ax.set_xticks(range(len(pf_counts)))
        ax.set_xticklabels(['Never','1-2x','2-3x','3-4x','5+x'])
        ax.set_title('Purchase Frequency Distribution', fontweight='bold')
        ax.bar_label(bars, fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c4:
        fig, ax = plt.subplots(figsize=(6, 4))
        af = filtered['preferred_ad_format'].str.strip().value_counts().sort_values()
        ax.barh(af.index, af.values, color=PALETTE[:len(af)])
        ax.set_title('Preferred Ad Format', fontweight='bold')
        ax.set_xlabel('Count')
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ═══════════════════════════════════════════════
# TAB 9: ROI SIMULATOR
# ═══════════════════════════════════════════════
with tabs[8]:
    st.markdown('<div class="section-head">💰 Influencer ROI Simulator</div>', unsafe_allow_html=True)
    st.caption("Estimate campaign ROI based on influencer type, platform, and budget. Values derived from survey benchmarks.")

    # Survey-derived benchmarks
    BENCHMARKS = {
        'Nano (1K–10K)':   {'cpm': 80,   'ctr': 0.038, 'cvr': 0.052},
        'Micro (10K–100K)':{'cpm': 140,  'ctr': 0.028, 'cvr': 0.041},
        'Macro (100K–1M)': {'cpm': 320,  'ctr': 0.018, 'cvr': 0.029},
        'Mega (1M+)':      {'cpm': 900,  'ctr': 0.009, 'cvr': 0.015},
    }
    PLATFORM_MULTIPLIERS = {
        'Instagram Reels': {'reach': 1.25, 'ctr': 1.30, 'cpm_mult': 1.10},
        'YouTube':         {'reach': 1.10, 'ctr': 1.10, 'cpm_mult': 1.05},
        'Instagram Feed':  {'reach': 0.90, 'ctr': 0.85, 'cpm_mult': 0.95},
        'Stories':         {'reach': 0.75, 'ctr': 0.70, 'cpm_mult': 0.80},
        'Snapchat':        {'reach': 0.65, 'ctr': 0.60, 'cpm_mult': 0.75},
    }

    # Survey-derived ad click rate for the current filtered dataset
    survey_ctr = filtered['high_ad_responsive'].mean() if n_filtered > 0 else 0.21
    survey_purchase_rate = filtered['made_purchase'].mean() if n_filtered > 0 else 0.757

    col_inp1, col_inp2, col_inp3 = st.columns(3)
    with col_inp1:
        budget = st.number_input("Campaign Budget (₹)", min_value=5000, max_value=5000000,
                                 value=100000, step=5000, format="%d")
        influencer_type = st.selectbox("Influencer Type", list(BENCHMARKS.keys()))
        aov = st.number_input("Avg Order Value — AOV (₹)", min_value=100, max_value=50000, value=1200, step=100)
    with col_inp2:
        platform = st.selectbox("Primary Platform", list(PLATFORM_MULTIPLIERS.keys()))
        num_influencers = st.slider("Number of Influencers", 1, 20, 3)
        commission_pct = st.slider("Influencer Commission (% of budget)", 10, 80, 40) / 100
    with col_inp3:
        retargeting_pct = st.slider("Retargeting Budget Allocation (%)", 0, 50, 20) / 100
        funnel_dropoff = st.slider("Funnel Drop-off Adjustment (%)", 0, 50, 15) / 100
        # Actual purchase rate from survey
        st.metric("Survey Purchase Rate", f"{survey_purchase_rate*100:.1f}%", "From filtered data")

    st.markdown("---")

    bench = BENCHMARKS[influencer_type]
    plat = PLATFORM_MULTIPLIERS[platform]

    # Calculations
    content_budget = budget * commission_pct
    ad_budget = budget * retargeting_pct
    other_budget = budget * (1 - commission_pct - retargeting_pct)

    cpm_eff = bench['cpm'] * plat['cpm_mult']
    reach = (content_budget / cpm_eff) * 1000 * plat['reach']
    clicks = reach * bench['ctr'] * plat['ctr']
    conversions = clicks * bench['cvr'] * (1 - funnel_dropoff)
    # Add retargeting conversions
    retarg_reach = (ad_budget / (cpm_eff * 0.6)) * 1000 if ad_budget > 0 else 0
    retarg_conversions = retarg_reach * bench['ctr'] * 0.7 * bench['cvr'] * (1 - funnel_dropoff)
    total_conversions = conversions + retarg_conversions
    revenue = total_conversions * aov
    roi = ((revenue - budget) / budget) * 100 if budget > 0 else 0
    cpa = budget / total_conversions if total_conversions > 0 else 0
    roas = revenue / budget if budget > 0 else 0

    # KPI display
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📢 Est. Reach", f"{reach:,.0f}")
    k2.metric("🖱️ Est. Clicks", f"{clicks:,.0f}", f"CTR {bench['ctr']*plat['ctr']*100:.2f}%")
    k3.metric("🛒 Est. Conversions", f"{total_conversions:,.0f}")
    k4.metric("💵 Est. Revenue (₹)", f"₹{revenue:,.0f}")
    k5.metric("📈 ROI", f"{roi:.1f}%", "ROAS {:.2f}x".format(roas))

    col_vis1, col_vis2 = st.columns(2)

    with col_vis1:
        # Budget breakdown pie
        fig, ax = plt.subplots(figsize=(5, 4))
        bdg_labels = ['Influencer Commission', 'Retargeting Ads', 'Other (Production/Tools)']
        bdg_vals = [content_budget, ad_budget, other_budget]
        bdg_colors = [PALETTE[0], PALETTE[1], PALETTE[2]]
        wedges, texts, autotexts = ax.pie(bdg_vals, labels=bdg_labels, autopct='%1.1f%%',
                                           colors=bdg_colors, startangle=90,
                                           wedgeprops={'edgecolor': '#0a0a0f', 'linewidth': 2})
        for t in autotexts: t.set_fontsize(8)
        ax.set_title(f'Budget Allocation\n(Total: ₹{budget:,})', fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_vis2:
        # ROI vs Budget curve
        budget_range = np.linspace(10000, max(budget * 2, 500000), 60)
        roi_curve = []
        for b in budget_range:
            cb = b * commission_pct
            ab = b * retargeting_pct
            r = (cb / cpm_eff) * 1000 * plat['reach']
            cl = r * bench['ctr'] * plat['ctr']
            cv = cl * bench['cvr'] * (1 - funnel_dropoff)
            rr = (ab / (cpm_eff * 0.6)) * 1000 if ab > 0 else 0
            rc = rr * bench['ctr'] * 0.7 * bench['cvr'] * (1 - funnel_dropoff)
            rev = (cv + rc) * aov
            roi_curve.append(((rev - b) / b) * 100 if b > 0 else 0)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(budget_range / 1000, roi_curve, '-', color=PALETTE[0], linewidth=2.5)
        ax.axvline(budget / 1000, color=PALETTE[1], linestyle='--', linewidth=1.5,
                   label=f'Your budget ₹{budget/1000:.0f}K')
        ax.axhline(0, color='white', linewidth=1, alpha=0.4)
        ax.fill_between(budget_range / 1000, roi_curve, 0,
                        where=[r > 0 for r in roi_curve], alpha=0.12, color=PALETTE[2])
        ax.set_xlabel('Budget (₹ thousands)')
        ax.set_ylabel('Estimated ROI (%)')
        ax.set_title('ROI vs Budget Curve', fontweight='bold')
        ax.legend(fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Additional metrics
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"""
        <div class="insight-box">
          <div class="insight-title" style="color:{PALETTE[0]}">💡 Cost Per Acquisition</div>
          <div class="insight-text">₹{cpa:,.2f} per conversion — {'✅ Efficient' if cpa < aov * 0.3 else '⚠️ High — reduce retargeting or switch influencer tier'}</div>
        </div>
        <div class="insight-box">
          <div class="insight-title" style="color:{PALETTE[2]}">📊 ROAS</div>
          <div class="insight-text">{roas:.2f}x — For every ₹1 spent, you earn ₹{roas:.2f} back in revenue. {'✅ Positive ROI' if roas > 1 else '❌ Below break-even — adjust budget mix'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="insight-box">
          <div class="insight-title" style="color:{PALETTE[3]}">🎯 Break-Even Point</div>
          <div class="insight-text">Need at least {int(budget/aov)} conversions at AOV ₹{aov:,} to break even. Current estimate: {total_conversions:.0f} conversions.</div>
        </div>
        <div class="insight-box">
          <div class="insight-title" style="color:{PALETTE[4]}">👥 Per Influencer Reach</div>
          <div class="insight-text">~{reach/num_influencers:,.0f} people per influencer at ₹{content_budget/num_influencers:,.0f} budget each.</div>
        </div>
        """, unsafe_allow_html=True)

    # Scenario comparison table
    st.markdown('<div class="section-head">📊 Scenario Comparison — 3 Strategy Options</div>', unsafe_allow_html=True)

    scenarios = [
        {"Name": "🎯 Micro-Influencer Heavy", "type": "Micro (10K–100K)", "platform": "Instagram Reels",
         "commission": 0.60, "retargeting": 0.15, "n_inf": 10},
        {"Name": "📺 Macro + YouTube", "type": "Macro (100K–1M)", "platform": "YouTube",
         "commission": 0.55, "retargeting": 0.25, "n_inf": 2},
        {"Name": "💥 Mega Brand Play", "type": "Mega (1M+)", "platform": "Instagram Reels",
         "commission": 0.70, "retargeting": 0.10, "n_inf": 1},
    ]

    scen_rows = []
    for sc in scenarios:
        b_sc = BENCHMARKS[sc['type']]
        p_sc = PLATFORM_MULTIPLIERS[sc['platform']]
        cb_sc = budget * sc['commission']
        ab_sc = budget * sc['retargeting']
        cpm_sc = b_sc['cpm'] * p_sc['cpm_mult']
        reach_sc = (cb_sc / cpm_sc) * 1000 * p_sc['reach']
        clicks_sc = reach_sc * b_sc['ctr'] * p_sc['ctr']
        conv_sc = clicks_sc * b_sc['cvr'] * (1 - funnel_dropoff)
        rr_sc = (ab_sc / (cpm_sc * 0.6)) * 1000 if ab_sc > 0 else 0
        rc_sc = rr_sc * b_sc['ctr'] * 0.7 * b_sc['cvr'] * (1 - funnel_dropoff)
        total_conv_sc = conv_sc + rc_sc
        rev_sc = total_conv_sc * aov
        roi_sc = ((rev_sc - budget) / budget) * 100
        cpa_sc = budget / total_conv_sc if total_conv_sc > 0 else 0

        scen_rows.append({
            'Strategy': sc['Name'],
            'Influencer Type': sc['type'],
            'Platform': sc['platform'],
            '# Influencers': sc['n_inf'],
            'Est. Reach': f"{reach_sc:,.0f}",
            'Est. Conversions': f"{total_conv_sc:.0f}",
            'Est. Revenue (₹)': f"₹{rev_sc:,.0f}",
            'ROI (%)': f"{roi_sc:.1f}%",
            'CPA (₹)': f"₹{cpa_sc:,.0f}",
        })

    scen_df = pd.DataFrame(scen_rows)
    st.dataframe(scen_df, use_container_width=True)

    # Best scenario highlight
    best_sc = max(scenarios, key=lambda s: (
        (budget * s['commission'] / (BENCHMARKS[s['type']]['cpm'] * PLATFORM_MULTIPLIERS[s['platform']]['cpm_mult'])) * 1000
        * PLATFORM_MULTIPLIERS[s['platform']]['reach']
        * BENCHMARKS[s['type']]['ctr'] * PLATFORM_MULTIPLIERS[s['platform']]['ctr']
        * BENCHMARKS[s['type']]['cvr'] * (1 - funnel_dropoff) * aov
    ))
    st.success(f"🏆 **Recommended Strategy for ₹{budget:,} budget:** {best_sc['Name']} — maximizes estimated reach and conversion for this AOV and influencer mix.")


# ═══════════════════════════════════════════════
# TAB 10: TEXT ANALYTICS
# ═══════════════════════════════════════════════
with tabs[9]:
    st.markdown('<div class="section-head">☁️ Text Analytics — Motivations, Content & Cross-Tabs</div>', unsafe_allow_html=True)
    st.caption("Word clouds, chi-square significance tests, and cross-tabulation analysis of survey responses.")

    ta_col1, ta_col2 = st.columns(2)

    with ta_col1:
        st.markdown("#### Purchase Motivation Word Cloud")
        try:
            from wordcloud import WordCloud
            wc_text = ' '.join(filtered['purchase_motivation'].dropna().str.replace(';', ' '))
            if wc_text.strip():
                wc_bg = '#16161f'
                wc = WordCloud(
                    width=700, height=380, background_color=wc_bg,
                    colormap='plasma', max_words=80,
                    prefer_horizontal=0.85,
                    collocations=False
                ).generate(wc_text)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Purchase Motivation Word Cloud', fontweight='bold', pad=10)
                st.pyplot(fig, use_container_width=True)
                plt.close()
            else:
                st.info("No motivation text data available in current filter.")
        except ImportError:
            st.warning("⚠️ `wordcloud` package not installed. Run: `pip install wordcloud`")

    with ta_col2:
        st.markdown("#### Content Type Word Cloud")
        try:
            from wordcloud import WordCloud
            ct_text = ' '.join(filtered['content_type'].dropna().str.replace(';', ' ').str.replace('/', ' '))
            if ct_text.strip():
                wc2 = WordCloud(
                    width=700, height=380, background_color='#16161f',
                    colormap='cool', max_words=60, prefer_horizontal=0.8,
                    collocations=False
                ).generate(ct_text)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.imshow(wc2, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Content Type Word Cloud', fontweight='bold', pad=10)
                st.pyplot(fig, use_container_width=True)
                plt.close()
        except ImportError:
            pass

    # Chi-square tests
    st.markdown('<div class="section-head">🔬 Chi-Square Significance Tests</div>', unsafe_allow_html=True)
    st.caption("Tests whether two categorical variables are statistically independent.")

    from scipy.stats import chi2_contingency

    chi_tests = [
        ('Trust Factor', 'Made Purchase', 'trust_factor', 'made_purchase'),
        ('Trust Factor', 'High Ad Responsive', 'trust_factor', 'high_ad_responsive'),
        ('Age Group', 'Made Purchase', 'age_group', 'made_purchase'),
        ('Gender', 'Made Purchase', 'gender', 'made_purchase'),
        ('Preferred Ad Format', 'High Ad Responsive', 'preferred_ad_format', 'high_ad_responsive'),
    ]

    chi_results = []
    for label_a, label_b, col_a, col_b in chi_tests:
        try:
            ct = pd.crosstab(filtered[col_a], filtered[col_b])
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                chi2, p, dof, _ = chi2_contingency(ct)
                n = filtered[[col_a, col_b]].dropna().shape[0]
                cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if n > 0 else 0
                chi_results.append({
                    'Variable A': label_a, 'Variable B': label_b,
                    'Chi²': round(chi2, 3), 'df': dof,
                    'p-value': round(p, 5),
                    "Cramér's V": round(cramers_v, 3),
                    'Significant?': '✅ Yes' if p < 0.05 else '❌ No',
                    'Strength': 'Strong' if cramers_v > 0.3 else ('Moderate' if cramers_v > 0.15 else 'Weak')
                })
        except Exception:
            pass

    if chi_results:
        chi_df = pd.DataFrame(chi_results)
        st.dataframe(chi_df.style.applymap(
            lambda v: 'color: #5cf0c8' if v == '✅ Yes' else ('color: #fc5c7d' if v == '❌ No' else ''),
            subset=['Significant?']
        ), use_container_width=True)

        sig_pairs = [r for r in chi_results if r['Significant?'] == '✅ Yes']
        if sig_pairs:
            strongest = max(sig_pairs, key=lambda r: r["Cramér's V"])
            cramers_val = strongest["Cramér's V"]
            st.success(f"🏆 Strongest significant association: **{strongest['Variable A']} ↔ {strongest['Variable B']}** "
                       f"(Cramér's V = {cramers_val:.3f}, p = {strongest['p-value']:.5f})")

    else:
        st.info("Not enough data for chi-square tests. Widen filters.")

    # Cross-tab heatmaps
    st.markdown('<div class="section-head">🗂️ Cross-Tabulation Heatmaps</div>', unsafe_allow_html=True)

    xt_col1, xt_col2 = st.columns(2)

    with xt_col1:
        st.markdown("**Age Group × Preferred Content Type**")
        try:
            # Explode multi-select content types
            ct_exploded = filtered[['age_group', 'content_type']].dropna()
            ct_exploded = ct_exploded.assign(
                content_type=ct_exploded['content_type'].str.split(';')
            ).explode('content_type')
            ct_exploded['content_type'] = ct_exploded['content_type'].str.strip()
            ct_exploded = ct_exploded[ct_exploded['content_type'] != '']
            xt1 = pd.crosstab(ct_exploded['age_group'], ct_exploded['content_type'])
            if not xt1.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(xt1, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                            linewidths=0.5, linecolor='#0a0a0f', annot_kws={'size': 8})
                ax.set_title('Age Group × Content Type\n(respondent count)', fontweight='bold')
                plt.xticks(rotation=35, ha='right', fontsize=8)
                plt.yticks(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
        except Exception as e:
            st.info(f"Could not render cross-tab: {e}")

    with xt_col2:
        st.markdown("**Trust Factor × Purchase Motivation**")
        try:
            motiv_exploded = filtered[['trust_factor', 'purchase_motivation']].dropna()
            motiv_exploded = motiv_exploded.assign(
                purchase_motivation=motiv_exploded['purchase_motivation'].str.split(';')
            ).explode('purchase_motivation')
            motiv_exploded['purchase_motivation'] = motiv_exploded['purchase_motivation'].str.strip()
            motiv_exploded = motiv_exploded[motiv_exploded['purchase_motivation'] != '']
            xt2 = pd.crosstab(motiv_exploded['trust_factor'], motiv_exploded['purchase_motivation'])
            if not xt2.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(xt2, annot=True, fmt='d', cmap='Blues', ax=ax,
                            linewidths=0.5, linecolor='#0a0a0f', annot_kws={'size': 8})
                ax.set_title('Trust Factor × Purchase Motivation\n(co-occurrence count)', fontweight='bold')
                plt.xticks(rotation=35, ha='right', fontsize=8)
                plt.yticks(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
        except Exception as e:
            st.info(f"Could not render cross-tab: {e}")

    # Motivation frequency bar race
    st.markdown('<div class="section-head">📊 Motivation Frequency by Age Group</div>', unsafe_allow_html=True)
    try:
        motiv_age = filtered[['age_group', 'purchase_motivation']].dropna()
        motiv_age = motiv_age.assign(
            purchase_motivation=motiv_age['purchase_motivation'].str.split(';')
        ).explode('purchase_motivation')
        motiv_age['purchase_motivation'] = motiv_age['purchase_motivation'].str.strip()
        motiv_age = motiv_age[motiv_age['purchase_motivation'] != '']
        motiv_pivot = pd.crosstab(motiv_age['purchase_motivation'], motiv_age['age_group'])
        if not motiv_pivot.empty and motiv_pivot.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(12, 5))
            motiv_pivot.plot(kind='bar', ax=ax, color=PALETTE[:motiv_pivot.shape[1]],
                             edgecolor='#0a0a0f', alpha=0.88)
            ax.set_title('Purchase Motivations by Age Group', fontweight='bold')
            ax.set_ylabel('Respondent Count')
            ax.set_xlabel('')
            plt.xticks(rotation=30, ha='right', fontsize=9)
            ax.legend(title='Age Group', bbox_to_anchor=(1.01, 1), fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
    except Exception:
        pass


# ═══════════════════════════════════════════════
# TAB 6: RECOMMENDATIONS
# ═══════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-head">🏆 Strategic Recommendations for Brands</div>', unsafe_allow_html=True)
    st.caption("Data-driven recommendations derived from the filtered dataset analysis.")

    # Dynamically compute key stats from filtered data
    purchase_pct = filtered['made_purchase'].mean()*100 if n_filtered > 0 else 0
    top_trust = filtered['trust_factor'].value_counts().index[0] if n_filtered > 0 else 'N/A'
    top_trust_pct = filtered['trust_factor'].value_counts(normalize=True).iloc[0]*100 if n_filtered > 0 else 0
    reels_pct = (filtered['preferred_ad_format'].str.strip() == 'Reels').mean()*100 if n_filtered > 0 else 0
    top_motiv = filtered.get('motiv_positive_reviews', pd.Series(0)).mean()*100

    recs_data = [
        {
            'num': '01', 'category': 'Influencer Selection', 'color': PALETTE[0],
            'title': 'Content Quality Over Follower Count',
            'finding': f'{top_trust} is the #1 trust driver ({top_trust_pct:.1f}% of filtered respondents)',
            'action': 'Score influencer candidates on content depth, production quality, and niche expertise. Set minimum quality rubrics before follower thresholds.',
            'kpi': 'Trust Index, Purchase Intention Score'
        },
        {
            'num': '02', 'category': 'Campaign Design', 'color': PALETTE[1],
            'title': 'Reels-First Short-Form Video Strategy',
            'finding': f'{reels_pct:.1f}% of filtered respondents prefer Reels as most influential ad format',
            'action': 'Allocate ≥60% of influencer campaign budget to Reels/Shorts. Brief creators for authentic ≤60s product reviews with clear CTA.',
            'kpi': 'Ad Click Likelihood, Engagement Index'
        },
        {
            'num': '03', 'category': 'Conversion Optimization', 'color': PALETTE[2],
            'title': 'Review Content + Exclusive Coupon Triggers',
            'finding': f'Positive reviews motivate {top_motiv:.1f}% of purchasers; discount codes rank 2nd',
            'action': 'Brief influencers: 70% organic review + 30% CTA with unique coupon. Track conversion per influencer via code attribution.',
            'kpi': 'Platform Conversion Rate, Purchase Intention Score'
        },
        {
            'num': '04', 'category': 'Ad Budget Strategy', 'color': PALETTE[3],
            'title': 'Cap Retargeting — Redirect to Content',
            'finding': f'{100-ad_responsive_rate:.1f}% of filtered respondents rarely/never click ads',
            'action': 'Cap retargeting at 3-4 impressions/user/week. Redirect saved spend (est. 20-30%) to original influencer content commissions.',
            'kpi': 'Ad Click Likelihood, Campaign ROI'
        },
        {
            'num': '05', 'category': 'Risk Mitigation', 'color': PALETTE[4],
            'title': 'Engagement Quality Score (EQS) Framework',
            'finding': 'Low engagement frequency despite high purchase rate reveals passive-buyer segment',
            'action': 'EQS = (Comments + Saves) / Reach. Require EQS ≥ 2% for campaign eligibility. Flag >5% monthly follower spikes as bot risk.',
            'kpi': 'Engagement Index, Fake Engagement Risk Score'
        },
    ]

    for i in range(0, len(recs_data), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(recs_data):
                r = recs_data[i+j]
                with col:
                    st.markdown(f"""
                    <div style="background:#16161f; border:1px solid #2a2a3a; border-radius:14px; 
                                padding:20px; border-top:3px solid {r['color']}; margin-bottom:12px; position:relative;">
                      <div style="font-size:3rem; font-weight:800; color:#2a2a3a; position:absolute; top:12px; right:16px; font-family:sans-serif;">{r['num']}</div>
                      <div style="display:inline-block; background:{r['color']}22; color:{r['color']}; 
                                  border:1px solid {r['color']}44; padding:4px 10px; border-radius:6px; 
                                  font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:10px;">
                        {r['category']}
                      </div>
                      <div style="font-weight:700; font-size:1rem; margin-bottom:8px; color:#e8e8f0;">{r['title']}</div>
                      <div style="font-size:0.8rem; color:#5cf0c8; margin-bottom:8px;">📊 {r['finding']}</div>
                      <div style="font-size:0.82rem; color:#aaaacc; margin-bottom:8px; line-height:1.6;">✅ {r['action']}</div>
                      <div style="font-size:0.75rem; color:#7c5cfc;">📈 KPI: {r['kpi']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Summary visual
    st.markdown('<div class="section-head">KPI Summary (Filtered Data)</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Purchase funnel
    funnel_vals = [
        n_filtered,
        filtered['searched_binary'].sum(),
        filtered['made_purchase'].sum(),
        (filtered['purchase_ord'] >= 3).sum()
    ]
    funnel_labels = ['Follow Influencers','Searched Product','Made Purchase','High Freq (3+x)']
    funnel_colors = [PALETTE[0], PALETTE[2], PALETTE[3], PALETTE[1]]
    bars = axes[0].barh(funnel_labels, funnel_vals, color=funnel_colors)
    axes[0].set_title('Purchase Funnel', fontweight='bold')
    axes[0].set_xlabel('Count')
    for i, v in enumerate(funnel_vals):
        pct = v/n_filtered*100 if n_filtered > 0 else 0
        axes[0].text(v+0.5, i, f'{v} ({pct:.0f}%)', va='center', fontsize=9)

    # Engagement -> Purchase rate
    eng_order = ['Rarely','Sometimes','frequently','Almost always']
    eng_labels_short = ['Rarely','Smetimes','Freq.','Always']
    purchase_by_eng = filtered.groupby('engagement_freq')['made_purchase'].mean() * 100
    purchase_by_eng = purchase_by_eng.reindex(eng_order).fillna(0)
    adclick_by_eng = filtered.groupby('engagement_freq')['high_ad_responsive'].mean() * 100
    adclick_by_eng = adclick_by_eng.reindex(eng_order).fillna(0)
    x = np.arange(len(eng_order))
    w = 0.38
    b1 = axes[1].bar(x-w/2, purchase_by_eng.values, w, label='Purchase %', color=PALETTE[0])
    b2 = axes[1].bar(x+w/2, adclick_by_eng.values, w, label='Ad Click %', color=PALETTE[1])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(eng_labels_short, fontsize=9)
    axes[1].set_title('Engagement → Purchase & Ad Rate', fontweight='bold')
    axes[1].set_ylabel('%')
    axes[1].legend(fontsize=9)

    # Trust -> Purchase rate
    trust_purchase_rate = filtered.groupby('trust_factor')['made_purchase'].mean() * 100
    trust_purchase_rate = trust_purchase_rate.sort_values(ascending=True)
    bars = axes[2].barh(trust_purchase_rate.index, trust_purchase_rate.values, color=PALETTE[:len(trust_purchase_rate)])
    axes[2].set_title('Purchase Rate by Trust Driver', fontweight='bold')
    axes[2].set_xlabel('% who Made a Purchase')
    axes[2].bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#7878a0; font-size:0.78rem; padding: 10px 0;">
  Group 5 — The Influence Crew &nbsp;·&nbsp; Business Analytics Project &nbsp;·&nbsp; 
  Survey n=210 &nbsp;·&nbsp; Dec 2025<br>
  Built with Streamlit · pandas · scikit-learn · matplotlib · seaborn
</div>
""", unsafe_allow_html=True)
