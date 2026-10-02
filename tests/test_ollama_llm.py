"""
Unit and Integration Tests for OllamaLLMService (Qwen3:8B).
Tests initialization, input handling, prompt formatting, output normalization,
mocked HTTP error states, and optional live Ollama integration.
"""

import os
import io
import json
import asyncio
import unittest
from unittest.mock import patch, MagicMock
import urllib.error

from signbridge.services.ollama_llm import (
    OllamaLLMService,
    OllamaLLMError,
    get_ollama_llm_service,
    DEFAULT_OLLAMA_API_URL,
    DEFAULT_OLLAMA_MODEL,
)


class TestOllamaLLMService(unittest.TestCase):

    def setUp(self):
        self.service = OllamaLLMService()

    def test_service_initialization_defaults(self):
        self.assertEqual(self.service.api_url, DEFAULT_OLLAMA_API_URL)
        self.assertEqual(self.service.model, DEFAULT_OLLAMA_MODEL)
        self.assertEqual(self.service.timeout, 30.0)
        self.assertEqual(self.service.generate_endpoint, f"{DEFAULT_OLLAMA_API_URL}/api/generate")

    def test_custom_config_values(self):
        custom_svc = OllamaLLMService(
            api_url="http://127.0.0.1:11434/",
            model="custom-qwen:8b",
            timeout=15.0
        )
        self.assertEqual(custom_svc.api_url, "http://127.0.0.1:11434")
        self.assertEqual(custom_svc.model, "custom-qwen:8b")
        self.assertEqual(custom_svc.timeout, 15.0)
        self.assertEqual(custom_svc.generate_endpoint, "http://127.0.0.1:11434/api/generate")

    def test_singleton_getter(self):
        svc1 = get_ollama_llm_service()
        svc2 = get_ollama_llm_service()
        self.assertIs(svc1, svc2)

    def test_empty_and_whitespace_input_handling(self):
        for empty_val in ["", "   ", "\n\t ", [], None]:
            res = self.service.generate_sentence_sync(empty_val)
            self.assertEqual(res["english"], "")
            self.assertEqual(res["glosses"], [])
            self.assertEqual(res["latency_ms"], 0.0)
            self.assertTrue(res["success"])

    def test_prompt_construction(self):
        prompt = self.service.construct_prompt(["I", "WANT", "WATER"])
        self.assertIn("ASL Glosses: I WANT WATER", prompt)
        self.assertIn("English Sentence:", prompt)

        prompt_clean = self.service.construct_prompt(["  hello12 ", "WORLD/IT "])
        self.assertIn("ASL Glosses: HELLO WORLD", prompt_clean)

    def test_output_cleanup_and_normalization(self):
        self.assertEqual(self.service.normalize_output('"i want water"'), "I want water.")
        self.assertEqual(self.service.normalize_output("<think>some reasoning</think> hello world"), "Hello world.")
        self.assertEqual(self.service.normalize_output("English Sentence: 'Thank you very much.'"), "Thank you very much.")
        self.assertEqual(self.service.normalize_output("   "), "")

    @patch("urllib.request.urlopen")
    def test_successful_ollama_response_mocked(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"response": "I want water."}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = self.service.generate_sentence_sync(["I", "WANT", "WATER"])
        self.assertEqual(res["english"], "I want water.")
        self.assertEqual(res["glosses"], ["I", "WANT", "WATER"])
        self.assertEqual(res["provider"], "ollama_qwen3")
        self.assertTrue(res["success"])
        self.assertGreaterEqual(res["latency_ms"], 0.0)

    @patch("urllib.request.urlopen")
    def test_malformed_ollama_json_response(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"NOT VALID JSON"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        with self.assertRaises(OllamaLLMError) as ctx:
            self.service.generate_sentence_sync("HELLO")
        self.assertIn("Malformed JSON", ctx.exception.message)
        self.assertEqual(ctx.exception.status_code, 500)

    @patch("urllib.request.urlopen")
    def test_ollama_unavailable_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with self.assertRaises(OllamaLLMError) as ctx:
            self.service.generate_sentence_sync("HELLO")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("unreachable", ctx.exception.message)
        self.assertTrue(ctx.exception.recoverable)

    @patch("urllib.request.urlopen")
    def test_empty_generated_response_handling(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"response": ""}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        with self.assertRaises(OllamaLLMError) as ctx:
            self.service.generate_sentence_sync("HELLO")
        self.assertIn("empty response", ctx.exception.message)
        self.assertEqual(ctx.exception.status_code, 500)

    @patch("urllib.request.urlopen")
    def test_async_generate_sentence(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"response": "I am going to school."}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(self.service.generate_sentence("I GO SCHOOL"))
            self.assertEqual(res["english"], "I am going to school.")
        finally:
            loop.close()

    @unittest.skipUnless(
        os.getenv("RUN_OLLAMA_INTEGRATION_TESTS") == "1",
        "Live Ollama integration test disabled by default (enable via RUN_OLLAMA_INTEGRATION_TESTS=1)"
    )
    def test_live_qwen3_integration(self):
        """Optional live integration test against actual running Ollama qwen3:8b instance."""
        res = self.service.generate_sentence_sync("I WANT WATER")
        self.assertTrue(res["success"])
        self.assertIn("water", res["english"].lower())
        self.assertGreater(res["latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
