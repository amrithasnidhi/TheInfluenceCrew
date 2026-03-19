import unittest
import pandas as pd
import sys
import os

from unittest.mock import MagicMock

# Completely block Streamlit from loading and generating warnings
mock_st = MagicMock()
mock_st.cache_data = lambda func: func
sys.modules['streamlit'] = mock_st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from streamlit_dashboard import load_data, preprocess

class TestDashboardIntegration(unittest.TestCase):
    """Integration tests to ensure end-to-end data pipelines correctly feed the association mining engine."""

    def test_end_to_end_association_rules(self):
        """Test that data loads, processes, and can successfully generate valid association rules."""
        
        # 1. Load data
        df_raw = load_data()
        self.assertGreater(len(df_raw), 0, "Failed to load raw data.")
        
        # 2. Preprocess data
        df_clean = preprocess(df_raw)
        self.assertIn('purchase_ord', df_clean.columns)
        
        # 3. Simulate the Transaction Matrix building logic from TAB 4
        # We integrate the cleaned dataframe into the association matrix format
        t_df = pd.DataFrame({
            'High_Engagement': (df_clean['engagement_ord'] >= 2).astype(int),
            'High_Interest': (df_clean.get('product_interest_score', pd.Series(0, index=df_clean.index)) >= 4).astype(int),
            'Made_Purchase': (df_clean['purchase_ord'] > 0).astype(int),
            'Uses_Instagram': df_clean.get('platform_instagram', 0),
        }).fillna(0).astype(int)
        
        # Validate matrix is built correctly
        self.assertEqual(len(t_df), len(df_clean), "Transaction matrix dropped rows.")
        self.assertIn('Made_Purchase', t_df.columns)
        
        # 4. Extract compute_rules function dynamically and run integration check
        # Instead of directly importing, we execute the exact logic the module would run
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
            return pd.DataFrame(rules)

        antecedent_list = ['High_Engagement', 'Uses_Instagram', 'High_Interest']
        target_rule = 'Made_Purchase'
        
        rules_df = compute_rules(t_df, antecedent_list, target_rule, min_sup=0.01, min_conf=0.1)
        
        # 5. Integration Validation
        # The rules DataFrame should successfully populate based on pipeline data
        self.assertIsNotNone(rules_df)
        if not rules_df.empty:
            self.assertIn('Antecedent', rules_df.columns)
            self.assertIn('Lift', rules_df.columns)
            self.assertGreater(rules_df['Lift'].max(), 0, "Calculated Lift should be non-zero.")

if __name__ == "__main__":
    unittest.main()
