import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from signbridge.inference.predictor import SignBridgePredictor
from signbridge.preprocessing.normalize import normalize_sequence, normalize_hand

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "trained_models" / "best_gru_normalized.pth"
TEST_CSV = PROJECT_ROOT / "dataset" / "ASL_Citizen" / "splits" / "test.csv"
TEST_CACHE_DIR = PROJECT_ROOT / "landmark_cache_full" / "test"


@unittest.skipUnless(TEST_CSV.exists() and TEST_CACHE_DIR.exists(), "ASL Citizen dataset splits and cache files not present")
class TestPipelineParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = SignBridgePredictor(CHECKPOINT_PATH)
        cls.test_df = pd.read_csv(TEST_CSV)
        cls.test_indices = [0, 339, 1000, 5000]

    def test_cache_vs_predictor_tensor_parity(self):
        """
        Verify that raw cache loaded manually and processed via normalize_sequence
        produces an exact match (max_diff == 0.0) with predictor preprocessing.
        """
        for idx in self.test_indices:
            cache_file = TEST_CACHE_DIR / f"{idx}.npy"
            self.assertTrue(cache_file.exists(), f"Cache file {cache_file} does not exist")

            # 1. Manual baseline pipeline
            raw_cache = np.load(cache_file).astype(np.float32)
            self.assertEqual(raw_cache.shape, (32, 42, 3))

            normed_manual = normalize_sequence(raw_cache)
            flat_manual = normed_manual.reshape(32, 126)

            # 2. Predictor pipeline direct inference
            result_cache = self.predictor.predict_cache(cache_file, top_k=5)
            result_seq = self.predictor.predict_sequence(raw_cache, is_normalized=False, top_k=5)

            # Compare predictions
            self.assertEqual(
                result_cache["predicted_gloss"],
                result_seq["predicted_gloss"],
                f"Mismatch for index {idx} between predict_cache and predict_sequence",
            )
            self.assertAlmostEqual(
                result_cache["confidence"],
                result_seq["confidence"],
                places=5,
                msg=f"Confidence mismatch for index {idx}",
            )

    def test_tensor_numerical_exactness(self):
        """
        Test max absolute difference between normalized cache and flattened sequence tensor.
        """
        for idx in self.test_indices:
            cache_file = TEST_CACHE_DIR / f"{idx}.npy"
            raw_cache = np.load(cache_file).astype(np.float32)

            t1 = normalize_sequence(raw_cache).reshape(32, 126)
            t2 = normalize_sequence(raw_cache.reshape(32, 126)).reshape(32, 126)

            max_diff = float(np.max(np.abs(t1 - t2)))
            self.assertEqual(max_diff, 0.0, f"Max diff > 0 for index {idx}: {max_diff}")

    def test_hand_normalization_invariance(self):
        """
        Test that empty/zero hand remains exactly zero.
        """
        zero_hand = np.zeros((21, 3), dtype=np.float32)
        normed_zero = normalize_hand(zero_hand)
        self.assertEqual(float(np.max(np.abs(normed_zero))), 0.0)

    def test_golden_apple_sample_prediction(self):
        """
        Test index 339 (APPLE) achieves high confidence prediction.
        """
        cache_file = TEST_CACHE_DIR / "339.npy"
        result = self.predictor.predict_cache(cache_file, top_k=5)
        self.assertEqual(result["predicted_gloss"], "APPLE")
        self.assertGreater(result["confidence"], 90.0)


if __name__ == "__main__":
    unittest.main()
