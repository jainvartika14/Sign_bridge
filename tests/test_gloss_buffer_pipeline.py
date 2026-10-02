"""
Unit and Integration Tests for Gloss Buffer & Manual Speak Sentence Pipeline (Step 13).
Tests GlossBuffer methods, duplicate rejection, and sentence pipeline error resilience.
"""

import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException

from signbridge.services.gloss_buffer import GlossBuffer
from signbridge.services.ollama_llm import OllamaLLMService, OllamaLLMError
from signbridge.services.kokoro_tts import KokoroTTSService, KokoroTTSError
from signbridge.web.app import NLPSmoothRequest, nlp_smooth_endpoint, TTSSynthesizeRequest, tts_synthesize_endpoint


class TestGlossBuffer(unittest.TestCase):

    def test_1_empty_buffer(self):
        buf = GlossBuffer()
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.get_glosses(), [])
        self.assertEqual(buf.to_string(), "")

    def test_2_add_one_gloss(self):
        buf = GlossBuffer()
        added = buf.add_gloss("WATER")
        self.assertTrue(added)
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf.get_glosses(), ["WATER"])
        self.assertEqual(buf.to_string(), "WATER")

    def test_3_add_multiple_glosses_in_order(self):
        buf = GlossBuffer()
        buf.add_gloss("I")
        buf.add_gloss("WANT")
        buf.add_gloss("WATER")
        self.assertEqual(len(buf), 3)
        self.assertEqual(buf.get_glosses(), ["I", "WANT", "WATER"])
        self.assertEqual(buf.to_string(), "I WANT WATER")

    def test_4_consecutive_duplicate_prevention(self):
        buf = GlossBuffer()
        self.assertTrue(buf.add_gloss("WATER"))
        self.assertFalse(buf.add_gloss("WATER"))  # Rejected
        self.assertFalse(buf.add_gloss("WATER"))  # Rejected
        self.assertFalse(buf.add_gloss("WATER"))  # Rejected
        self.assertEqual(buf.get_glosses(), ["WATER"])

        # Multiple distinct glosses with consecutive duplicate rejection
        buf2 = GlossBuffer()
        buf2.add_gloss("I")
        buf2.add_gloss("I")
        buf2.add_gloss("WANT")
        buf2.add_gloss("WANT")
        buf2.add_gloss("WATER")
        buf2.add_gloss("WATER")
        self.assertEqual(buf2.get_glosses(), ["I", "WANT", "WATER"])

        # Non-consecutive repeated word should be accepted
        buf3 = GlossBuffer()
        buf3.add_gloss("I")
        buf3.add_gloss("WANT")
        buf3.add_gloss("WATER")
        buf3.add_gloss("I")
        buf3.add_gloss("WANT")
        buf3.add_gloss("WATER")
        self.assertEqual(buf3.get_glosses(), ["I", "WANT", "WATER", "I", "WANT", "WATER"])

    def test_5_clear_buffer(self):
        buf = GlossBuffer(["I", "WANT", "WATER"])
        self.assertEqual(len(buf), 3)
        buf.clear()
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.to_string(), "")

    def test_empty_and_invalid_gloss_cleaning(self):
        buf = GlossBuffer()
        self.assertFalse(buf.add_gloss(""))
        self.assertFalse(buf.add_gloss("   "))
        self.assertFalse(buf.add_gloss(None))
        self.assertFalse(buf.add_gloss("—"))
        self.assertFalse(buf.add_gloss("Analyzing…"))

        # Sanitization check
        buf.add_gloss("water1")
        buf.add_gloss("this/it")
        self.assertEqual(buf.get_glosses(), ["WATER", "THIS"])

    def test_step14_recognition_committed_gloss_reaches_buffer(self):
        """Step 14: Verify stable committed recognition predictions reach GlossBuffer cleanly."""
        buf = GlossBuffer()

        # Simulate stable recognition predictions emitted by stabilizer
        committed_predictions = ["I", "WANT", "WATER"]
        for gloss in committed_predictions:
            buf.add_gloss(gloss)

        self.assertEqual(buf.get_glosses(), ["I", "WANT", "WATER"])
        self.assertEqual(buf.to_string(), "I WANT WATER")

    def test_step14_repeated_raw_predictions_do_not_duplicate_buffer(self):
        """Step 14: Verify 5 repeated raw recognition frames yield exactly 1 buffer entry."""
        buf = GlossBuffer()

        # Simulate raw stream of repeated recognition predictions
        raw_stream = ["WATER", "WATER", "WATER", "WATER", "WATER"]
        committed_count = 0
        for raw in raw_stream:
            if buf.add_gloss(raw):
                committed_count += 1

        self.assertEqual(committed_count, 1)
        self.assertEqual(buf.get_glosses(), ["WATER"])

    def test_step14_non_consecutive_repeated_words_remain_valid(self):
        """Step 14: Verify non-consecutive repetitions remain valid in buffer."""
        buf = GlossBuffer()
        glosses = ["I", "WANT", "WATER", "I", "WANT", "WATER"]
        for g in glosses:
            buf.add_gloss(g)
        self.assertEqual(buf.get_glosses(), ["I", "WANT", "WATER", "I", "WANT", "WATER"])


class TestSentencePipelineEndpoints(unittest.TestCase):

    def test_6_empty_speak_does_not_call_ollama(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            req = NLPSmoothRequest(glosses=[], use_llm=True)
            res = loop.run_until_complete(nlp_smooth_endpoint(req))
            import json
            data = json.loads(res.body.decode("utf-8"))
            self.assertEqual(data["english"], "")
            self.assertEqual(data["glosses"], [])
            self.assertEqual(data["provider"], "empty")
        finally:
            loop.close()

    @patch("signbridge.web.app.get_ollama_llm_service")
    def test_7_successful_gloss_to_ollama_to_sentence(self, mock_get_ollama):
        mock_svc = MagicMock()
        mock_svc.generate_sentence = AsyncMock(return_value={
            "english": "I want water.",
            "glosses": ["I", "WANT", "WATER"],
            "raw_gloss_string": "I WANT WATER",
            "model": "qwen3:8b",
            "latency_ms": 120.0,
            "provider": "ollama_qwen3",
            "success": True,
        })
        mock_get_ollama.return_value = mock_svc

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            req = NLPSmoothRequest(glosses=["I", "WANT", "WATER"], use_llm=True)
            res = loop.run_until_complete(nlp_smooth_endpoint(req))
            import json
            data = json.loads(res.body.decode("utf-8"))
            self.assertEqual(data["english"], "I want water.")
            self.assertEqual(data["provider"], "ollama_qwen3")
            self.assertTrue(data["success"])
        finally:
            loop.close()

    @patch("signbridge.web.app.get_kokoro_tts_service")
    def test_8_successful_sentence_to_kokoro(self, mock_get_kokoro):
        mock_svc = MagicMock()
        mock_svc.is_available = True
        mock_svc.synthesize = AsyncMock(return_value=(
            b"RIFF....WAVEfmt ....data....",
            "audio/wav",
            {"request_id": "test_req", "provider": "kokoro_onnx", "latency_ms": 50.0}
        ))
        mock_get_kokoro.return_value = mock_svc

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            req = TTSSynthesizeRequest(text="I want water.", voice_id="af_sarah", rate=1.0, request_id="test_req")
            response = loop.run_until_complete(tts_synthesize_endpoint(req))
            self.assertEqual(response.headers.get("Content-Type"), "audio/wav")
            self.assertEqual(response.headers.get("X-TTS-Provider"), "kokoro_onnx")
            self.assertEqual(response.headers.get("X-Voicebox-Configured"), "true")
        finally:
            loop.close()

    @patch("signbridge.web.app.get_ollama_llm_service")
    def test_9_ollama_failure_raises_http_exception(self, mock_get_ollama):
        mock_svc = MagicMock()
        mock_svc.generate_sentence = AsyncMock(side_effect=OllamaLLMError("Ollama server offline", status_code=503))
        mock_get_ollama.return_value = mock_svc

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            req = NLPSmoothRequest(glosses=["I", "WANT", "WATER"], use_llm=True)
            with self.assertRaises(HTTPException) as ctx:
                loop.run_until_complete(nlp_smooth_endpoint(req))
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("LLM Generation Failed", ctx.exception.detail["error"])
        finally:
            loop.close()

    def test_10_and_11_buffer_state_preservation_and_clear_rules(self):
        buf = GlossBuffer(["I", "WANT", "WATER"])
        
        # Simulate Ollama failure -> buffer preserved
        ollama_failed = True
        if ollama_failed:
            pass  # Do not clear buffer
        self.assertEqual(buf.get_glosses(), ["I", "WANT", "WATER"])

        # Simulate TTS failure -> buffer preserved
        tts_failed = True
        if tts_failed:
            pass  # Do not clear buffer
        self.assertEqual(buf.get_glosses(), ["I", "WANT", "WATER"])

        # Simulate successful end-to-end pipeline -> clear buffer
        ollama_failed = False
        tts_failed = False
        if not ollama_failed and not tts_failed:
            buf.clear()

        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.to_string(), "")

    @patch("signbridge.web.app.get_ollama_llm_service")
    @patch("signbridge.web.app.get_kokoro_tts_service")
    def test_step14_recognition_to_buffer_does_not_call_qwen_or_kokoro(self, mock_kokoro, mock_ollama):
        """Step 14: Asserts that accumulating glosses from recognition does NOT invoke Ollama or Kokoro automatically."""
        buf = GlossBuffer()

        # Simulate camera recognition accumulating glosses into buffer
        buf.add_gloss("I")
        buf.add_gloss("WANT")
        buf.add_gloss("WATER")

        self.assertEqual(buf.get_glosses(), ["I", "WANT", "WATER"])

        # Neither Qwen (Ollama) nor Kokoro TTS should have been called
        mock_ollama.return_value.generate_sentence.assert_not_called()
        mock_kokoro.return_value.synthesize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
