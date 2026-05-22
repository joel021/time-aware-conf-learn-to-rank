import unittest
import os
import shutil
import sys

# Ensure root directory is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

class TestMainFlow(unittest.TestCase):

    def setUp(self):
        # Determine dataset availability
        self.db_dir = os.path.join(root_dir, "data", "ml-100k")
        if not os.path.exists(self.db_dir) or not os.path.exists(os.path.join(self.db_dir, "info.json")):
            self.skipTest("ml-100k dataset is not available")

        # Cleanup existing runs directories for ml-100k and mf combination
        self.cleanup_runs()

    def cleanup_runs(self):
        runs_dir = os.path.join(root_dir, "runs")
        if os.path.exists(runs_dir):
            # Delete directories starting with ml-100k-mf-
            for name in os.listdir(runs_dir):
                path = os.path.join(runs_dir, name)
                if os.path.isdir(path) and name.startswith("ml-100k-mf"):
                    shutil.rmtree(path)
            
            # Delete splits folder for ml-100k
            splits_unlabeled = os.path.join(runs_dir, "data_splits", "unlabeled", "ml-100k")
            if os.path.exists(splits_unlabeled):
                shutil.rmtree(splits_unlabeled)
            
            splits_labeled = os.path.join(runs_dir, "data_splits", "labeled", "ml-100k")
            if os.path.exists(splits_labeled):
                shutil.rmtree(splits_labeled)

    def test_execute_main_flow(self):
        try:
            from main import run_k_folds
        except ImportError:
            self.skipTest("Could not import run_k_folds from main")

        setups = {
            "ml-100k-learn-rank-mf-test": {
                "database_name": "ml-100k",
                "model_name": "mf",
                "batch_size": 1024,
                "min_inter_per_user": 100
            }
        }
        
        # Run the full k-fold cross validation flow (we run from split_position=3 with k_folds=5 to be fast)
        run_k_folds(setups, split_position=3, k=5)

        # Verify that the expected files were generated
        runs_dir = os.path.join(root_dir, "runs")
        self.assertTrue(os.path.exists(runs_dir), "runs directory should be created")
        
        # Check that fold 3 directories and files were generated
        fold_dir = os.path.join(runs_dir, "ml-100k-mf-3")
        self.assertTrue(os.path.exists(fold_dir), f"Directory {fold_dir} should be created")
        
        # Verify specific expected files exist
        self.assertTrue(os.path.exists(os.path.join(fold_dir, "model-3.pth")), "model-3.pth should exist")
        self.assertTrue(
            os.path.exists(os.path.join(fold_dir, "history-3.json")) or 
            os.path.exists(os.path.join(fold_dir, "history-3standard.json")), 
            "history json file should exist"
        )
        self.assertTrue(os.path.exists(os.path.join(fold_dir, "test_error_conf-3.csv")), "test_error_conf-3.csv should exist")
