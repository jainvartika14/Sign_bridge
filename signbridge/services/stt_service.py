"""
Speech-to-Text (STT) Service for SignBridge.
Coordinates server-side speech recognition, audio stream transcription,
and mapping of hearing-partner spoken English to recognized ASL glosses.
"""

import os
import io
import re
import uuid
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
import urllib.request
import urllib.error
import json
import asyncio

logger = logging.getLogger("signbridge.stt")

# Environment configuration
DEFAULT_STT_PROVIDER = os.getenv("STT_PROVIDER", "webspeech_primary")
DEFAULT_STT_API_URL = os.getenv("STT_API_URL", "")
DEFAULT_STT_API_KEY = os.getenv("STT_API_KEY", "")
DEFAULT_STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en-US")


class STTServiceError(Exception):
    """Base exception for STT service errors."""
    def __init__(self, message: str, status_code: int = 500, recoverable: bool = False):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.recoverable = recoverable


class STTService:
    """
    Speech-to-Text service coordinator.
    Handles server-side audio transcription and ASL gloss vocabulary mapping.
    """

    # Common English-to-ASL lemmatization roots for conversational signs
    LEMMA_MAP = {
        "eating": "EAT", "ate": "EAT", "eats": "EAT",
        "going": "GO", "went": "GO", "goes": "GO",
        "seeing": "SEE", "saw": "SEE", "sees": "SEE",
        "wanting": "WANT", "wants": "WANT", "wanted": "WANT",
        "liking": "LIKE", "likes": "LIKE", "liked": "LIKE",
        "drinking": "DRINK", "drank": "DRINK", "drinks": "DRINK",
        "apples": "APPLE", "bananas": "BANANA", "books": "BOOK",
        "cars": "CAR", "cats": "CAT", "dogs": "DOG",
        "friends": "FRIEND", "families": "FAMILY",
        "helping": "HELP", "helped": "HELP", "helps": "HELP",
        "running": "RUN", "ran": "RUN", "runs": "RUN",
        "working": "WORK", "worked": "WORK", "works": "WORK",
        "learning": "LEARN", "learned": "LEARN", "learns": "LEARN",
        "thanks": "THANKYOU", "thank": "THANKYOU",
        "sorry": "SORRY", "hello": "HELLO", "hi": "HELLO",
        "goodbye": "GOODBYE", "bye": "GOODBYE", "please": "PLEASE",
        "yes": "YES", "yeah": "YES", "no": "NO", "nope": "NO",
    }

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        language: Optional[str] = None,
    ):
        self.api_url = api_url or DEFAULT_STT_API_URL
        self.api_key = api_key or DEFAULT_STT_API_KEY
        self.language = language or DEFAULT_STT_LANGUAGE

    @property
    def is_cloud_configured(self) -> bool:
        """Checks if external cloud STT credentials are set."""
        return bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"))

    def extract_asl_glosses(self, text: str, vocabulary: Optional[Set[str]] = None) -> List[Dict[str, str]]:
        """
        Extract ASL gloss matches from spoken English transcript.
        Maps spoken words and idioms to corresponding signs recognized by SignBridge.
        """
        if not text:
            return []

        words = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()).split()
        matches = []
        seen = set()

        for word in words:
            # 1. Direct match (e.g. "apple" -> "APPLE")
            upper = word.upper()
            if vocabulary is None or upper in vocabulary:
                if upper not in seen:
                    matches.append({"word": word, "gloss": upper, "type": "direct"})
                    seen.add(upper)
                    continue

            # 2. Lemmatized match (e.g. "eating" -> "EAT")
            if word in self.LEMMA_MAP:
                gloss = self.LEMMA_MAP[word]
                if vocabulary is None or gloss in vocabulary:
                    if gloss not in seen:
                        matches.append({"word": word, "gloss": gloss, "type": "stemmed"})
                        seen.add(gloss)
                        continue

            # 3. Simple suffix stripping
            stem = None
            if word.endswith("ing") and len(word) > 5:
                stem = word[:-3].upper()
            elif word.endswith("ed") and len(word) > 4:
                stem = word[:-2].upper()
            elif word.endswith("s") and len(word) > 3:
                stem = word[:-1].upper()

            if stem and (vocabulary is None or stem in vocabulary) and stem not in seen:
                matches.append({"word": word, "gloss": stem, "type": "stemmed"})
                seen.add(stem)

        return matches

    async def transcribe_audio_chunk(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
        vocabulary: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe an uploaded audio chunk (fallback when browser STT is unsupported).
        """
        start_time = time.perf_counter()
        req_id = str(uuid.uuid4())

        if not audio_bytes or len(audio_bytes) < 44:
            raise STTServiceError("Audio payload is empty or invalid header.", status_code=400)

        # If external cloud STT is configured, send request
        if self.is_cloud_configured and self.api_url:
            try:
                transcript, is_final, conf = await self._call_cloud_stt(audio_bytes, content_type)
            except Exception as e:
                logger.error(f"[STT:{req_id}] Cloud STT error: {e}")
                raise STTServiceError(f"Cloud STT failed: {str(e)}", status_code=502, recoverable=True)
        else:
            # Informative message for server-side chunk without external cloud provider
            transcript = "Server audio chunk received. (Use Chrome/Edge for instant Web Speech STT, or configure STT_API_KEY in .env)"
            is_final = True
            conf = 0.85

        matched_glosses = self.extract_asl_glosses(transcript, vocabulary)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "request_id": req_id,
            "transcript": transcript,
            "is_final": is_final,
            "confidence": conf,
            "matched_glosses": matched_glosses,
            "latency_ms": latency_ms,
            "cloud_configured": self.is_cloud_configured,
        }

    async def _call_cloud_stt(self, audio_bytes: bytes, content_type: str) -> Tuple[str, bool, float]:
        """Make HTTP POST to external cloud STT endpoint."""
        loop = asyncio.get_event_loop()
        headers = {
            "Content-Type": content_type,
            "Authorization": f"Bearer {self.api_key}",
        }

        def _request():
            req = urllib.request.Request(self.api_url, data=audio_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                transcript = data.get("text", "") or data.get("transcript", "")
                is_final = data.get("is_final", True)
                conf = float(data.get("confidence", 0.9))
                return transcript, is_final, conf

        return await loop.run_in_executor(None, _request)


# Global Singleton Instance
_stt_service: Optional[STTService] = None


def get_stt_service() -> STTService:
    """Retrieve or create the global STT service singleton."""
    global _stt_service
    if _stt_service is None:
        _stt_service = STTService()
    return _stt_service
