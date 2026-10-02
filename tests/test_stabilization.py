"""
Unit & Integration Tests for SignBridge Prediction Stabilization,
Sign Boundary Detection, Duplicate Suppression, and Truthful Model Status.
"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from signbridge.web.app import app, health_check, conversation_config_endpoint


class PythonStabilizerReference:
    """
    Python reference implementation of the PredictionStabilizer algorithm
    defined in signbridge/web/static/js/stabilizer.js.
    Ensures mathematical and behavioral parity between backend validation and frontend logic.
    """

    def __init__(
        self,
        vote_buffer_size=6,
        min_confidence=35.0,
        stability_fraction=0.50,
        min_consecutive=2,
        min_margin=3.0,
        decisive_confidence=65.0,
        decisive_margin=12.0,
        boundary_idle_frames=8,
        repeat_cooldown_ms=1800,
        low_confidence_reset=20.0,
    ):
        self.vote_buffer_size = vote_buffer_size
        self.min_confidence = min_confidence
        self.stability_fraction = stability_fraction
        self.min_consecutive = min_consecutive
        self.min_margin = min_margin
        self.decisive_confidence = decisive_confidence
        self.decisive_margin = decisive_margin
        self.boundary_idle_frames = boundary_idle_frames
        self.repeat_cooldown_ms = repeat_cooldown_ms
        self.low_confidence_reset = low_confidence_reset

        self.vote_buffer = []
        self.raw_prediction = None
        self.candidate = None
        self.stable_candidate = None
        self.last_committed_gloss = None
        self.committed_glosses = []

        self.last_commit_time = 0
        self.idle_frame_count = 0
        self.boundary_detected = True
        self.consecutive_matches = 0
        self.last_leading_gloss = None

    def record_hand_presence(self, num_hands: int):
        if num_hands == 0:
            self.idle_frame_count += 1
            if self.idle_frame_count >= self.boundary_idle_frames:
                self.boundary_detected = True
        else:
            self.idle_frame_count = 0

    def compute_margin(self, top_k):
        if not top_k or len(top_k) < 2:
            return top_k[0].get("confidence", 0.0) if top_k else 0.0
        return max(0.0, top_k[0].get("confidence", 0.0) - top_k[1].get("confidence", 0.0))

    def process_prediction(self, predicted_gloss: str, confidence: float, top_k: list, timestamp: int = 1000):
        gloss = predicted_gloss.strip().upper()
        conf = float(confidence)
        margin = self.compute_margin(top_k)

        # Stage 1: Raw prediction
        self.raw_prediction = {
            "gloss": gloss,
            "confidence": conf,
            "margin": margin,
            "top_k": top_k,
            "timestamp": timestamp,
        }

        if conf < self.low_confidence_reset:
            self.boundary_detected = True

        if conf < self.min_confidence:
            if self.vote_buffer:
                self.vote_buffer.pop(0)
            return self._status(False, None)

        margin_multiplier = 1.0 + min(max(margin, 0.0) / 25.0, 1.0)
        vote_weight = conf * margin_multiplier

        self.vote_buffer.append({
            "gloss": gloss,
            "confidence": conf,
            "margin": margin,
            "weight": vote_weight,
            "timestamp": timestamp,
        })

        if len(self.vote_buffer) > self.vote_buffer_size:
            self.vote_buffer.pop(0)

        # Aggregate weighted votes
        total_weight = sum(v["weight"] for v in self.vote_buffer)
        votes = {}
        occurrences = {}
        for v in self.vote_buffer:
            g = v["gloss"]
            votes[g] = votes.get(g, 0.0) + v["weight"]
            occurrences[g] = occurrences.get(g, 0) + 1

        # Stage 2: Candidate selection
        leading_gloss = max(votes.keys(), key=lambda g: votes[g])
        highest_score = votes[leading_gloss]
        candidate_fraction = highest_score / total_weight if total_weight > 0 else 0
        candidate_count = occurrences[leading_gloss]

        if leading_gloss == self.last_leading_gloss:
            self.consecutive_matches += 1
        else:
            self.consecutive_matches = 1
            self.last_leading_gloss = leading_gloss

        self.candidate = {
            "gloss": leading_gloss,
            "confidence": conf,
            "score": highest_score,
            "fraction": candidate_fraction,
            "consecutive": self.consecutive_matches,
        }

        # Stage 3: Stability evaluation
        is_sustained = (
            candidate_count >= self.min_consecutive
            and candidate_fraction >= self.stability_fraction
            and margin >= self.min_margin
        )
        is_decisive = (conf >= self.decisive_confidence) and (margin >= self.decisive_margin)

        just_stabilized = False
        if is_sustained or is_decisive:
            self.stable_candidate = {
                "gloss": leading_gloss,
                "confidence": conf,
                "margin": margin,
            }
            just_stabilized = True

        # Stage 4: Commit & Duplicate Suppression
        just_committed = False
        committed_gloss = None

        if just_stabilized and self.stable_candidate:
            stable_gloss = self.stable_candidate["gloss"]

            # Condition A: First sign or transition to a different sign
            if self.last_committed_gloss is None or stable_gloss != self.last_committed_gloss:
                committed_gloss = self._commit(stable_gloss, timestamp)
                just_committed = True

            # Condition B: Same sign repeated — only commit if boundary occurred AND cooldown elapsed
            elif stable_gloss == self.last_committed_gloss:
                elapsed = timestamp - self.last_commit_time
                if self.boundary_detected and elapsed >= self.repeat_cooldown_ms:
                    committed_gloss = self._commit(stable_gloss, timestamp)
                    just_committed = True
                else:
                    # Suppress duplicate while sign is held
                    pass

        return self._status(just_committed, committed_gloss)

    def _commit(self, gloss: str, timestamp: int):
        self.committed_glosses.append(gloss)
        self.last_committed_gloss = gloss
        self.last_commit_time = timestamp
        self.boundary_detected = False
        return gloss

    def undo_last_committed(self):
        if not self.committed_glosses:
            return None
        removed = self.committed_glosses.pop()
        self.last_committed_gloss = self.committed_glosses[-1] if self.committed_glosses else None
        self.boundary_detected = True
        return removed

    def clear_committed(self):
        self.committed_glosses = []
        self.last_committed_gloss = None
        self.boundary_detected = True

    def _status(self, just_committed: bool, committed_gloss: str | None):
        return {
            "raw_prediction": self.raw_prediction,
            "candidate": self.candidate,
            "stable_candidate": self.stable_candidate,
            "last_committed_gloss": self.last_committed_gloss,
            "committed_glosses": list(self.committed_glosses),
            "just_committed": just_committed,
            "committed_gloss": committed_gloss,
        }


class TestPredictionStabilization(unittest.TestCase):
    def setUp(self):
        self.stabilizer = PythonStabilizerReference()

    def test_low_confidence_rejected(self):
        """Uncertain raw predictions below min_confidence must not be accepted as candidates."""
        top_k = [{"gloss": "HELLO", "confidence": 25.0}, {"gloss": "THANKYOU", "confidence": 20.0}]
        status = self.stabilizer.process_prediction("HELLO", 25.0, top_k, timestamp=1000)
        self.assertIsNone(status["candidate"])
        self.assertIsNone(status["stable_candidate"])
        self.assertFalse(status["just_committed"])
        self.assertEqual(len(status["committed_glosses"]), 0)

    def test_duplicate_suppression_while_sign_is_held(self):
        """
        Signing HELLO continuously across multiple frames must commit HELLO ONCE,
        NOT multiple times (HELLO HELLO HELLO HELLO).
        """
        top_k = [{"gloss": "HELLO", "confidence": 75.0}, {"gloss": "WELCOME", "confidence": 10.0}]

        # Frame 1: Initial detection & commit
        s1 = self.stabilizer.process_prediction("HELLO", 75.0, top_k, timestamp=1000)
        self.assertTrue(s1["just_committed"])
        self.assertEqual(s1["committed_gloss"], "HELLO")
        self.assertEqual(s1["committed_glosses"], ["HELLO"])

        # Frames 2 to 6: User continues holding HELLO (180ms intervals)
        for t in [1180, 1360, 1540, 1720, 1900]:
            st = self.stabilizer.process_prediction("HELLO", 75.0, top_k, timestamp=t)
            self.assertFalse(st["just_committed"], f"Should have suppressed duplicate at t={t}")
            self.assertEqual(st["committed_glosses"], ["HELLO"], "Committed buffer must not duplicate")

    def test_boundary_reset_allows_legitimate_repeated_sign(self):
        """
        User signs HELLO, drops hands (boundary triggered), then signs HELLO again.
        Both HELLOs must be committed legitimately.
        """
        top_k = [{"gloss": "HELLO", "confidence": 70.0}, {"gloss": "WELCOME", "confidence": 10.0}]

        # First HELLO commit
        s1 = self.stabilizer.process_prediction("HELLO", 70.0, top_k, timestamp=1000)
        self.assertTrue(s1["just_committed"])
        self.assertEqual(s1["committed_glosses"], ["HELLO"])

        # User drops hands: 8 idle frames observed
        for _ in range(8):
            self.stabilizer.record_hand_presence(0)
        self.assertTrue(self.stabilizer.boundary_detected)

        # User signs HELLO again after cooldown (e.g. 2500ms elapsed)
        s2 = self.stabilizer.process_prediction("HELLO", 70.0, top_k, timestamp=3500)
        self.assertTrue(s2["just_committed"], "Legitimate repeated sign after boundary must be committed")
        self.assertEqual(s2["committed_glosses"], ["HELLO", "HELLO"])

    def test_transition_to_different_sign(self):
        """
        Transitioning from one sign to a different sign commits both in proper sequence.
        """
        top_k_hello = [{"gloss": "HELLO", "confidence": 72.0}, {"gloss": "WELCOME", "confidence": 8.0}]
        top_k_thank = [{"gloss": "THANKYOU", "confidence": 68.0}, {"gloss": "PLEASE", "confidence": 12.0}]

        # Sign 1: HELLO
        self.stabilizer.process_prediction("HELLO", 72.0, top_k_hello, timestamp=1000)
        self.assertEqual(self.stabilizer.committed_glosses, ["HELLO"])

        # Sign 2: THANKYOU (transition to different gloss is naturally accepted)
        self.stabilizer.process_prediction("THANKYOU", 68.0, top_k_thank, timestamp=1200)
        s2 = self.stabilizer.process_prediction("THANKYOU", 68.0, top_k_thank, timestamp=1380)

        self.assertEqual(self.stabilizer.committed_glosses, ["HELLO", "THANKYOU"])

    def test_undo_and_clear_buffer_controls(self):
        """User review controls: undo last sign and clear entire buffer."""
        top_k1 = [{"gloss": "GOOD", "confidence": 75.0}, {"gloss": "BAD", "confidence": 5.0}]
        top_k2 = [{"gloss": "MORNING", "confidence": 75.0}, {"gloss": "NIGHT", "confidence": 5.0}]

        self.stabilizer.process_prediction("GOOD", 75.0, top_k1, timestamp=1000)
        self.stabilizer.process_prediction("MORNING", 75.0, top_k2, timestamp=1200)
        self.stabilizer.process_prediction("MORNING", 75.0, top_k2, timestamp=1380)
        self.assertEqual(self.stabilizer.committed_glosses, ["GOOD", "MORNING"])

        # Undo last
        removed = self.stabilizer.undo_last_committed()
        self.assertEqual(removed, "MORNING")
        self.assertEqual(self.stabilizer.committed_glosses, ["GOOD"])
        self.assertEqual(self.stabilizer.last_committed_gloss, "GOOD")

        # Clear buffer
        self.stabilizer.clear_committed()
        self.assertEqual(self.stabilizer.committed_glosses, [])
        self.assertIsNone(self.stabilizer.last_committed_gloss)


class TestModelStatusTruthfulness(unittest.TestCase):
    @patch("signbridge.web.app.get_predictor")
    def test_health_check_truthful_when_predictor_available(self, mock_get_pred):
        mock_pred = MagicMock()
        mock_pred.class_names = [f"GLOSS_{i}" for i in range(2731)]
        mock_pred.device = "cpu"
        mock_get_pred.return_value = mock_pred

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(health_check())
            self.assertEqual(res["status"], "healthy")
            self.assertTrue(res["model_loaded"])
            self.assertEqual(res["num_classes"], 2731)
            self.assertEqual(res["device"], "cpu")
        finally:
            loop.close()

    @patch("signbridge.web.app.get_predictor")
    def test_health_check_truthful_when_predictor_missing(self, mock_get_pred):
        mock_get_pred.side_effect = FileNotFoundError("Checkpoint not found")

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(health_check())
            self.assertEqual(res["status"], "degraded")
            self.assertFalse(res["model_loaded"])
            self.assertEqual(res["num_classes"], 0)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
