"""
Focused Unit Tests for SignBridge Kokoro TTS Service.
Tests model initialization, reusable singleton loading, request validation,
voice fallback, speaking rate clamping, and WAV synthesis output.
"""

import unittest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

from signbridge.services.kokoro_tts import (
    KokoroTTSService,
    KokoroTTSError,
    get_kokoro_tts_service,
)


class TestKokoroTTSService(unittest.TestCase):
    def setUp(self):
        self.service = KokoroTTSService()

    def test_service_initialization_and_availability(self):
        """Verify Kokoro service initializes and correctly reports model availability."""
        self.assertTrue(self.service.is_available, "Kokoro model should be available in verified environment.")
        voices = self.service.get_available_voices()
        self.assertIsInstance(voices, list)
        self.assertGreater(len(voices), 0)
        self.assertTrue(any(v["id"] == "af_sarah" for v in voices))

    def test_singleton_getter(self):
        """Verify get_kokoro_tts_service returns the same reusable instance."""
        s1 = get_kokoro_tts_service()
        s2 = get_kokoro_tts_service()
        self.assertIs(s1, s2, "Singleton getter should return identical instance.")

    def test_validation_empty_text_raises_error(self):
        """Empty or whitespace-only text must raise KokoroTTSError 400."""
        with self.assertRaises(KokoroTTSError) as ctx:
            self.service.validate_request("")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertFalse(ctx.exception.recoverable)

        with self.assertRaises(KokoroTTSError) as ctx:
            self.service.validate_request("   \n\t  ")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_validation_rate_clamping(self):
        """Speaking rate should clamp cleanly between 0.5x and 2.0x."""
        _, _, rate_low = self.service.validate_request("Hello", rate=0.1)
        self.assertEqual(rate_low, 0.5)

        _, _, rate_high = self.service.validate_request("Hello", rate=4.0)
        self.assertEqual(rate_high, 2.0)

        _, _, rate_normal = self.service.validate_request("Hello", rate=1.2)
        self.assertEqual(rate_normal, 1.2)

    def test_validation_voice_fallback(self):
        """Unrecognized voice_id should fallback to default af_sarah."""
        _, voice, _ = self.service.validate_request("Hello", voice_id="invalid_voice_name")
        self.assertEqual(voice, self.service.default_voice_id)

    def test_synthesize_valid_text(self):
        """Synthesize text into valid WAV audio bytes and verify metadata."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            audio_bytes, media_type, meta = loop.run_until_complete(
                self.service.synthesize(
                    text="Testing Kokoro TTS service implementation.",
                    voice_id="af_sarah",
                    rate=1.0,
                    request_id="test_kokoro_req_001",
                )
            )

            self.assertEqual(media_type, "audio/wav")
            self.assertIsInstance(audio_bytes, bytes)
            self.assertGreater(len(audio_bytes), 1000)

            # Check RIFF header for valid WAV file format
            self.assertEqual(audio_bytes[:4], b"RIFF")
            self.assertEqual(audio_bytes[8:12], b"WAVE")

            self.assertEqual(meta["request_id"], "test_kokoro_req_001")
            self.assertEqual(meta["provider"], "kokoro_onnx")
            self.assertEqual(meta["voice"], "af_sarah")
            self.assertEqual(meta["sample_rate"], 24000)
            self.assertIn("latency_ms", meta)
            self.assertGreater(meta["latency_ms"], 0)
        finally:
            loop.close()

    def test_synthesize_unavailable_model_raises_error(self):
        """When model is not available, synthesize must raise 503 error."""
        unavail_service = KokoroTTSService(
            model_path=Path("nonexistent_model.onnx"),
            voices_path=Path("nonexistent_voices.bin"),
        )
        self.assertFalse(unavail_service.is_available)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with self.assertRaises(KokoroTTSError) as ctx:
                loop.run_until_complete(unavail_service.synthesize("Hello"))
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertTrue(ctx.exception.recoverable)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
