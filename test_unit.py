import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# Completely block Streamlit from loading and generating warnings
mock_st = MagicMock()
mock_st.cache_data = lambda func: func
sys.modules['streamlit'] = mock_st

# Import dashboard functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from streamlit_dashboard import preprocess

class TestDataPreprocessingUnit(unittest.TestCase):
    """Unit tests for the data preprocessing maps and edge cases."""
    
    def setUp(self):
        """Set up a deterministic, edge-case heavy raw dataframe."""
        self.raw_data = pd.DataFrame({
            'ID': [1, 2, 3],
            'Age Group': ['13-18', '40+', 'Unknown'],
            'Average Monthly Online Spending (₹)': ['< ₹500', '₹5000+', None],
            'Which platforms do you regularly use to consume influencer content': ['Instagram;YouTube', 'Facebook', ''],
            'How often have you purchased a product because of an influencer recommendation?': ['Never', 'More than 5 times', 'Invalid']
        })

    def test_preprocess_mappings(self):
        """Test if the deterministic data maps correctly to expected ordinal/binary values."""
        df_clean = preprocess(self.raw_data)
        
        # Check spending_ord mapping
        self.assertEqual(df_clean.loc[0, 'spending_ord'], 1) # < ₹500
        self.assertEqual(df_clean.loc[1, 'spending_ord'], 4) # ₹5000+
        self.assertTrue(pd.isna(df_clean.loc[2, 'spending_ord']) or df_clean.loc[2, 'spending_ord'] == 2) # fallback
        
        # Check purchase mapping ('Never' -> 0, 'More than 5 times' -> 4)
        self.assertEqual(df_clean.loc[0, 'purchase_ord'], 0)
        self.assertEqual(df_clean.loc[1, 'purchase_ord'], 4)
        
        # Check derived 'made_purchase' boolean mapping
        self.assertEqual(df_clean.loc[0, 'made_purchase'], 0)
        self.assertEqual(df_clean.loc[1, 'made_purchase'], 1)

    def test_platform_splitting(self):
        """Test if the platform binary encoding correctly splits semicolon-separated platforms."""
        df_clean = preprocess(self.raw_data)
        
        # First user has Instagram and YouTube
        self.assertEqual(df_clean.loc[0, 'platform_instagram'], 1)
        self.assertEqual(df_clean.loc[0, 'platform_youtube'], 1)
        self.assertEqual(df_clean.loc[0, 'platform_facebook'], 0)
        
        # Second user has Facebook only
        self.assertEqual(df_clean.loc[1, 'platform_instagram'], 0)
        self.assertEqual(df_clean.loc[1, 'platform_facebook'], 1)

if __name__ == "__main__":
    unittest.main()
