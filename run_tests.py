"""
run_tests.py
Professional Test Runner for The Influence Crew Dashboard
Provides clean, verbosity-enabled output formatting and suppresses noisy dashboard warnings.
"""
import unittest
import logging
import sys

# Suppress noisy Streamlit warnings when importing UI modules inside a testing terminal
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.elements.pyplot").setLevel(logging.ERROR)

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 INITIALIZING: The Influence Crew - Professional Test Suite")
    print("=" * 70)
    
    # Discover and run all test modules in the directory automatically
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='.', pattern='test_*.py')
    
    # Use TextTestRunner with verbosity=2 to detail which exact tests run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ SUCCESS: All pipelines, statistical parameters, and ML validations passed.")
        sys.exit(0)
    else:
        print("❌ FAILED: Please review the test stacktraces above to address regressions.")
        sys.exit(1)
