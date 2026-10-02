"""
Unit & Integration Tests for SignBridge Two-Way Real-Time Conversation Mode.
Tests STT Service and API endpoint handlers directly.
"""

import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException

from signbridge.web.app import (
    TTSSynthesizeRequest,
    tts_synthesize_endpoint,
    tts_voices_endpoint,
    conversation_config_endpoint,
    stt_transcribe_endpoint,
)
from signbridge.services.stt_service import STTService, STTServiceError


class TestSTTService(unittest.TestCase):
    def setUp(self):
        self.stt = STTService()
        self.test_vocab = {"APPLE", "BANANA", "EAT", "HELP", "THANKYOU", "PLEASE", "HELLO"}

    def test_extract_asl_glosses_direct_match(self):
        matches = self.stt.extract_asl_glosses("I want an apple please", self.test_vocab)
        glosses = [m["gloss"] for m in matches]
        self.assertIn("APPLE", glosses)
        self.assertIn("PLEASE", glosses)

    def test_extract_asl_glosses_lemmatization(self):
        matches = self.stt.extract_asl_glosses("She was eating apples", self.test_vocab)
        glosses = [m["gloss"] for m in matches]
        self.assertIn("EAT", glosses)
        self.assertIn("APPLE", glosses)

    def test_transcribe_audio_empty_payload_raises(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with self.assertRaises(STTServiceError) as ctx:
                loop.run_until_complete(self.stt.transcribe_audio_chunk(b"short"))
            self.assertEqual(ctx.exception.status_code, 400)
        finally:
            loop.close()


class TestConversationAPIEndpoints(unittest.TestCase):
    def test_get_conversation_config(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(conversation_config_endpoint())
            self.assertIn("tts", res)
            self.assertEqual(res["tts"]["provider"], "kokoro_onnx")
            self.assertEqual(res["tts"]["default_voice"], "af_sarah")
            self.assertIn("stt", res)
            self.assertIn("mode_support", res)
            self.assertIn("two_way", res["mode_support"])
        finally:
            loop.close()

    def test_get_tts_voices(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(tts_voices_endpoint())
            self.assertEqual(res["provider"], "kokoro_onnx")
            self.assertEqual(res["default_voice"], "af_sarah")
            self.assertIn("voices", res)
            self.assertGreater(len(res["voices"]), 0)
        finally:
            loop.close()

    def test_post_tts_synthesize_endpoint_success(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            req = TTSSynthesizeRequest(
                text="Hello, thank you for meeting with me.",
                voice_id="af_sarah",
                rate=1.0,
                request_id="test_req_kokoro_001",
            )
            response = loop.run_until_complete(tts_synthesize_endpoint(req))
            self.assertEqual(response.headers.get("Content-Type"), "audio/wav")
            self.assertEqual(response.headers.get("X-Request-ID"), "test_req_kokoro_001")
            self.assertEqual(response.headers.get("X-TTS-Provider"), "kokoro_onnx")
            self.assertEqual(response.headers.get("X-Voicebox-Configured"), "true")
            self.assertIn("X-TTS-Latency-MS", response.headers)
            self.assertTrue(len(response.body) > 44)
            self.assertEqual(response.body[:4], b"RIFF")
            self.assertEqual(response.body[8:12], b"WAVE")
        finally:
            loop.close()

    def test_post_tts_synthesize_empty_fails(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            req = TTSSynthesizeRequest(text="   ")
            with self.assertRaises(HTTPException) as ctx:
                loop.run_until_complete(tts_synthesize_endpoint(req))
            self.assertEqual(ctx.exception.status_code, 400)
        finally:
            loop.close()

    def test_post_tts_synthesize_invalid_voice_fallback(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            req = TTSSynthesizeRequest(
                text="Hello from fallback voice test.",
                voice_id="non_existent_voice_999",
                rate=1.0,
                request_id="test_req_fallback",
            )
            response = loop.run_until_complete(tts_synthesize_endpoint(req))
            self.assertEqual(response.headers.get("Content-Type"), "audio/wav")
            self.assertEqual(response.headers.get("X-TTS-Provider"), "kokoro_onnx")
            self.assertTrue(len(response.body) > 44)
            self.assertEqual(response.body[:4], b"RIFF")
        finally:
            loop.close()

    def test_post_tts_synthesize_failure_handling(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from unittest.mock import patch
            from signbridge.services.kokoro_tts import KokoroTTSError

            with patch("signbridge.web.app.get_kokoro_tts_service") as mock_get_svc:
                mock_svc = MagicMock()
                mock_svc.synthesize = AsyncMock(side_effect=KokoroTTSError("Synthesis engine failed", status_code=500, request_id="fail_req"))
                mock_get_svc.return_value = mock_svc

                req = TTSSynthesizeRequest(text="Fail test", request_id="fail_req")
                with self.assertRaises(HTTPException) as ctx:
                    loop.run_until_complete(tts_synthesize_endpoint(req))
                self.assertEqual(ctx.exception.status_code, 500)
                self.assertIn("Synthesis engine failed", ctx.exception.detail["error"])
        finally:
            loop.close()

    def test_post_stt_transcribe_chunk_endpoint(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Create a minimal 44-byte WAV header
            wav_header = (
                b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
                b"\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
            )
            mock_upload = AsyncMock()
            mock_upload.read.return_value = wav_header
            mock_upload.content_type = "audio/wav"

            res = loop.run_until_complete(stt_transcribe_endpoint(mock_upload))
            # res is a JSONResponse
            import json
            data = json.loads(res.body.decode("utf-8"))
            self.assertIn("transcript", data)
            self.assertIn("is_final", data)
            self.assertIn("confidence", data)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
