"""
Ollama Local LLM Service for SignBridge (Qwen3:8B Integration).
Translates raw ASL gloss sequences into fluent, natural English sentences
using local Qwen3:8B neural language model via Ollama REST API.
"""

import os
import re
import json
import time
import socket
import logging
import urllib.request
import urllib.error
import asyncio
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger("signbridge.ollama_llm")

# Environment configuration defaults
DEFAULT_OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434").rstrip("/")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
DEFAULT_OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30.0"))

SYSTEM_PROMPT = (
    "You are an ASL gloss to English translator. "
    "Convert the input ASL glosses into one short, natural English sentence. "
    "Do not provide explanations, analysis, or alternatives. "
    "Do not invent people, locations, emotions, or events. "
    "Output ONLY the final English sentence."
)


class OllamaLLMError(Exception):
    """Base exception for Ollama LLM service errors."""
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        recoverable: bool = True,
        raw_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.recoverable = recoverable
        self.raw_error = raw_error


class OllamaLLMService:
    """
    Reusable local Ollama LLM service for SignBridge.
    Communicates with local Ollama HTTP REST endpoints to translate ASL glosses to English sentences.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_url = (api_url or DEFAULT_OLLAMA_API_URL).rstrip("/")
        self.model = model or DEFAULT_OLLAMA_MODEL
        self.timeout = timeout if timeout is not None else DEFAULT_OLLAMA_TIMEOUT
        self.system_prompt = SYSTEM_PROMPT

    @property
    def generate_endpoint(self) -> str:
        return f"{self.api_url}/api/generate"

    @property
    def tags_endpoint(self) -> str:
        return f"{self.api_url}/api/tags"

    def is_available(self) -> bool:
        """
        Check if the Ollama server is reachable and the configured model is installed.
        """
        try:
            req = urllib.request.Request(self.tags_endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                    model_base = self.model.split(":")[0]
                    return model_base in models or any(self.model in m.get("name", "") for m in data.get("models", []))
            return False
        except Exception as e:
            logger.debug(f"[OllamaLLM] Availability check failed: {e}")
            return False

    def construct_prompt(self, glosses: List[str]) -> str:
        """
        Format clean gloss tokens into prompt payload.
        """
        cleaned_tokens = [
            g.strip().upper().rstrip("0123456789").removesuffix("/IT")
            for g in glosses if g and g.strip()
        ]
        gloss_str = " ".join(cleaned_tokens)
        return f"ASL Glosses: {gloss_str}\nEnglish Sentence:"

    def normalize_output(self, raw_text: str) -> str:
        """
        Clean and format model response output.
        Strips reasoning blocks, surrounding quotes, and leading explanatory text.
        """
        if not raw_text:
            return ""

        # Remove thinking/reasoning blocks if present in string
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
        cleaned = cleaned.strip().strip('"').strip("'").strip()

        # Remove prefix like 'English Sentence:' if model repeated prompt
        if cleaned.lower().startswith("english sentence:"):
            cleaned = cleaned[17:].strip().strip('"').strip("'").strip()

        # Collapse whitespace
        cleaned = " ".join(cleaned.split())

        # Enforce capitalization & ending punctuation if not empty
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
            if cleaned[-1] not in ".!?":
                cleaned += "."

        return cleaned

    def generate_sentence_sync(self, glosses: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Synchronously convert ASL gloss sequence into natural English sentence via Ollama API.
        """
        if not glosses:
            gloss_tokens = []
        elif isinstance(glosses, str):
            gloss_tokens = [g.strip() for g in glosses.split() if g.strip()]
        else:
            gloss_tokens = [str(g).strip() for g in glosses if g and str(g).strip()]

        cleaned_glosses = [g.upper().rstrip("0123456789").removesuffix("/IT") for g in gloss_tokens if g]
        raw_gloss_string = " ".join(cleaned_glosses)

        # Handle empty/whitespace input safely
        if not cleaned_glosses:
            return {
                "english": "",
                "glosses": [],
                "raw_gloss_string": "",
                "model": self.model,
                "latency_ms": 0.0,
                "provider": "ollama_qwen3",
                "success": True,
            }

        prompt_str = self.construct_prompt(cleaned_glosses)
        payload = {
            "model": self.model,
            "system": self.system_prompt,
            "prompt": prompt_str,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 256,
            }
        }

        req_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.generate_endpoint,
            data=req_bytes,
            headers={"Content-Type": "application/json"}
        )

        start_time = time.perf_counter()

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                if resp.status != 200:
                    raise OllamaLLMError(
                        f"Ollama server returned HTTP {resp.status}",
                        status_code=resp.status,
                        recoverable=True
                    )

                res_body = resp.read().decode("utf-8")
                try:
                    res_json = json.loads(res_body)
                except json.JSONDecodeError as pe:
                    raise OllamaLLMError(
                        "Malformed JSON response from Ollama server",
                        status_code=500,
                        recoverable=False,
                        raw_error=pe
                    )

                raw_response = res_json.get("response", "")
                if not raw_response and "thinking" in res_json:
                    raw_response = res_json.get("thinking", "")

                english_sentence = self.normalize_output(raw_response)

                if not english_sentence:
                    raise OllamaLLMError(
                        "Ollama model generated an empty response.",
                        status_code=500,
                        recoverable=True
                    )

                return {
                    "english": english_sentence,
                    "glosses": cleaned_glosses,
                    "raw_gloss_string": raw_gloss_string,
                    "model": self.model,
                    "latency_ms": elapsed_ms,
                    "provider": "ollama_qwen3",
                    "success": True,
                }

        except urllib.error.HTTPError as he:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            if he.code == 404:
                raise OllamaLLMError(
                    f"Ollama model '{self.model}' not found on server.",
                    status_code=404,
                    recoverable=False,
                    raw_error=he
                )
            raise OllamaLLMError(
                f"Ollama HTTP error: {he.code} - {he.reason}",
                status_code=he.code,
                recoverable=True,
                raw_error=he
            )
        except (urllib.error.URLError, socket.error, socket.timeout) as ne:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            raise OllamaLLMError(
                f"Ollama server unreachable at {self.api_url}: {str(ne)}",
                status_code=503,
                recoverable=True,
                raw_error=ne
            )
        except OllamaLLMError:
            raise
        except Exception as ge:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            raise OllamaLLMError(
                f"Unexpected Ollama LLM error: {str(ge)}",
                status_code=500,
                recoverable=True,
                raw_error=ge
            )

    async def generate_sentence(self, glosses: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Asynchronously convert ASL gloss sequence into natural English sentence using asyncio thread executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate_sentence_sync, glosses)


# Global singleton instance
_ollama_llm_service_instance: Optional[OllamaLLMService] = None


def get_ollama_llm_service() -> OllamaLLMService:
    """
    Returns the persistent global OllamaLLMService singleton instance.
    """
    global _ollama_llm_service_instance
    if _ollama_llm_service_instance is None:
        _ollama_llm_service_instance = OllamaLLMService()
    return _ollama_llm_service_instance
