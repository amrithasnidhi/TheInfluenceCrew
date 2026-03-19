import os
import sys
import unittest
import pandas as pd
from unittest.mock import MagicMock

# Completely block Streamlit from loading and generating warnings
mock_st = MagicMock()
mock_st.cache_data = lambda func: func
sys.modules['streamlit'] = mock_st

# Mock matplotlib
import matplotlib.pyplot as plt
fig_mock = MagicMock()
ax_mock = MagicMock()
ax_mock.pie.return_value = (MagicMock(), MagicMock(), MagicMock())
plt.subplots = MagicMock(return_value=(fig_mock, ax_mock))
plt.bar = MagicMock()
plt.barh = MagicMock()
plt.plot = MagicMock()

# Mock seaborn
sys.modules['seaborn'] = MagicMock()

# Import dashboard functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from streamlit_dashboard import load_data, preprocess

class TestDataPipelineRegression(unittest.TestCase):
    """Regression tests for the data preprocessing pipeline to ensure the baseline models function properly."""
    
    def test_load_data_baseline(self):
        """Test if data loads with baseline characteristics (at least 210 records)."""
        df = load_data()
        self.assertIsNotNone(df)
        self.assertGreaterEqual(len(df), 210, "Regression Alert: Loaded data size is smaller than baseline.")
        self.assertIn('Age Group', df.columns)
        self.assertIn('Gender', df.columns)
        
    def test_preprocess_regression(self):
        """Test if data preprocessing creates the required downstream columns without regressions."""
        df_raw = load_data()
        df_clean = preprocess(df_raw)
        
        expected_cols = [
            'spending_ord', 'daily_time_ord', 'engagement_ord', 'purchase_ord',
            'ad_click_ord', 'age_ord', 'searched_binary', 'ad_influenced_binary',
            'high_ad_responsive', 'made_purchase', 'platform_instagram', 'platform_youtube'
        ]
        
        for col in expected_cols:
            self.assertIn(col, df_clean.columns, f"Regression Alert: Missing expected preprocessing column - {col}")
            
        # Specific business logic assertions for regressions
        self.assertTrue(all(df_clean['high_ad_responsive'].isin([0, 1])))
        self.assertTrue(all(df_clean['made_purchase'].isin([0, 1])))

    def test_model_training_regression(self):
        """Test if the generated dataset still supports baseline model training metrics."""
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        
        df_raw = load_data()
        df = preprocess(df_raw)
        
        feat_cols = ['spending_ord', 'daily_time_ord', 'engagement_ord', 'product_interest_score',
                     'satisfaction_score', 'ad_click_ord', 'retargeting_score',
                     'searched_binary', 'ad_influenced_binary']
                     
        # Ensure all feature columns exist
        for col in feat_cols:
            self.assertIn(col, df.columns, f"Regression Alert: Missing feature column - {col}")
            
        model_df = df[feat_cols + ['high_ad_responsive', 'purchase_ord']].dropna()
        self.assertGreaterEqual(len(model_df), 30, "Regression Alert: Not enough data for modeling.")
        
        X = model_df[feat_cols].values
        y_ad = model_df['high_ad_responsive'].values
        y_pur = model_df['purchase_ord'].values
        
        sc_m = StandardScaler()
        X_sc = sc_m.fit_transform(X)
        
        # Logistic Regression Test
        lr = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        lr.fit(X_sc, y_ad)
        y_prob = lr.predict_proba(X_sc)[:, 1]
        auc = roc_auc_score(y_ad, y_prob)
        self.assertGreaterEqual(auc, 0.60, f"Regression Alert: Logistic Regression baseline AUC dropped to {auc:.3f}")
        
        # Random Forest Test
        rf = RandomForestClassifier(n_estimators=80, max_depth=4, random_state=42)
        rf.fit(X_sc, y_ad)
        importances = rf.feature_importances_
        self.assertEqual(len(importances), len(feat_cols), "Regression Alert: RF Feature importance length mismatch.")
        self.assertGreater(max(importances), 0.05, "Regression Alert: Max feature importance too low, model may be broken.")
        
        # Linear Regression Test
        reg = LinearRegression()
        reg.fit(X_sc, y_pur)
        r2 = reg.score(X_sc, y_pur)
        self.assertGreaterEqual(r2, 0.10, f"Regression Alert: Linear Regression baseline R2 dropped to {r2:.3f}")

if __name__ == "__main__":
    unittest.main()
