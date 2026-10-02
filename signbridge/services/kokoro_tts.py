"""
Kokoro Text-to-Speech (TTS) Service for SignBridge.
Provides low-latency, local neural speech synthesis using Kokoro ONNX model and soundfile.
"""

import os
import io
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import soundfile as sf

try:
    from kokoro_onnx import Kokoro
    _KOKORO_PACKAGE_AVAILABLE = True
except ImportError:
    _KOKORO_PACKAGE_AVAILABLE = False
    Kokoro = None

logger = logging.getLogger("signbridge.kokoro_tts")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_KOKORO_MODEL_PATH = PROJECT_ROOT / "models" / "kokoro" / "kokoro-v0_19.onnx"
DEFAULT_KOKORO_VOICES_PATH = PROJECT_ROOT / "models" / "kokoro" / "voices.bin"
DEFAULT_KOKORO_VOICE_ID = os.getenv("KOKORO_VOICE_ID", "af_sarah")


class KokoroTTSError(Exception):
    """Base exception for Kokoro TTS service errors."""
    def __init__(self, message: str, status_code: int = 500, recoverable: bool = False, request_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.recoverable = recoverable
        self.request_id = request_id


class KokoroTTSService:
    """
    Reusable local Kokoro TTS Service.
    Loads and reuses the Kokoro ONNX model instance across synthesis requests.
    """

    # Human-friendly labels for standard Kokoro voices
    VOICE_PROFILES = {
        "af_sarah": {"name": "American Female (Sarah)", "lang": "en-us", "gender": "female"},
        "af_bella": {"name": "American Female (Bella)", "lang": "en-us", "gender": "female"},
        "af_nicole": {"name": "American Female (Nicole)", "lang": "en-us", "gender": "female"},
        "af_sky": {"name": "American Female (Sky)", "lang": "en-us", "gender": "female"},
        "am_adam": {"name": "American Male (Adam)", "lang": "en-us", "gender": "male"},
        "am_michael": {"name": "American Male (Michael)", "lang": "en-us", "gender": "male"},
        "bf_emma": {"name": "British Female (Emma)", "lang": "en-gb", "gender": "female"},
        "bf_isabella": {"name": "British Female (Isabella)", "lang": "en-gb", "gender": "female"},
        "bm_george": {"name": "British Male (George)", "lang": "en-gb", "gender": "male"},
        "bm_lewis": {"name": "British Male (Lewis)", "lang": "en-gb", "gender": "male"},
    }

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        voices_path: Optional[Path | str] = None,
        default_voice_id: Optional[str] = None,
    ):
        self.model_path = Path(model_path or DEFAULT_KOKORO_MODEL_PATH)
        self.voices_path = Path(voices_path or DEFAULT_KOKORO_VOICES_PATH)
        self.default_voice_id = default_voice_id or DEFAULT_KOKORO_VOICE_ID

        self._kokoro_instance: Optional[Kokoro] = None
        self._load_error: Optional[str] = None
        self._init_model()

    def _init_model(self):
        """Initialize the reusable Kokoro ONNX model instance once."""
        if not _KOKORO_PACKAGE_AVAILABLE:
            self._load_error = "kokoro_onnx Python package is not installed."
            logger.warning(f"[KokoroTTS] {self._load_error}")
            return

        if not self.model_path.exists():
            self._load_error = f"Kokoro model file not found at {self.model_path}"
            logger.warning(f"[KokoroTTS] {self._load_error}")
            return

        if not self.voices_path.exists():
            self._load_error = f"Kokoro voices asset file not found at {self.voices_path}"
            logger.warning(f"[KokoroTTS] {self._load_error}")
            return

        try:
            logger.info(f"[KokoroTTS] Initializing Kokoro ONNX model from {self.model_path}...")
            self._kokoro_instance = Kokoro(str(self.model_path), str(self.voices_path))
            self._load_error = None
            logger.info(f"[KokoroTTS] Model loaded successfully. Available voices: {len(self.get_raw_voices())}")
        except Exception as e:
            self._load_error = f"Failed to initialize Kokoro model: {str(e)}"
            logger.error(f"[KokoroTTS] {self._load_error}")
            self._kokoro_instance = None

    @property
    def is_available(self) -> bool:
        """Return True if Kokoro ONNX model is initialized and ready for synthesis."""
        return self._kokoro_instance is not None

    def get_raw_voices(self) -> List[str]:
        """Return list of voice identifiers available in the loaded voice asset."""
        if not self.is_available:
            return []
        try:
            return self._kokoro_instance.get_voices()
        except Exception:
            return []

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Return structured metadata list for supported Kokoro voices."""
        raw_voices = self.get_raw_voices()
        if not raw_voices:
            return []

        result = []
        for voice_id in raw_voices:
            profile = self.VOICE_PROFILES.get(voice_id, {
                "name": voice_id.replace("_", " ").title(),
                "lang": "en-us",
                "gender": "neutral",
            })
            result.append({
                "id": voice_id,
                "name": profile["name"],
                "lang": profile["lang"],
                "gender": profile["gender"],
                "is_default": (voice_id == self.default_voice_id),
            })
        return result

    def validate_request(self, text: str, voice_id: Optional[str] = None, rate: float = 1.0) -> Tuple[str, str, float]:
        """Validate and sanitize synthesis parameters."""
        if not text or not text.strip():
            raise KokoroTTSError("Text payload cannot be empty.", status_code=400, recoverable=False)

        sanitized_text = text.strip()
        if len(sanitized_text) > 2000:
            sanitized_text = sanitized_text[:2000]

        selected_voice = (voice_id or self.default_voice_id).strip()
        available = set(self.get_raw_voices())

        if available and selected_voice not in available:
            logger.warning(f"[KokoroTTS] Voice '{selected_voice}' not available; falling back to '{self.default_voice_id}'")
            selected_voice = self.default_voice_id if self.default_voice_id in available else list(available)[0]

        clamped_rate = max(0.5, min(2.0, float(rate)))
        return sanitized_text, selected_voice, clamped_rate

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        rate: float = 1.0,
        request_id: Optional[str] = None,
    ) -> Tuple[bytes, str, Dict[str, Any]]:
        """
        Synthesize text into natural WAV audio bytes using local Kokoro ONNX model.

        Returns:
            Tuple of (audio_bytes, media_type, metadata_dict)
        """
        req_id = request_id or str(uuid.uuid4())

        if not self.is_available:
            raise KokoroTTSError(
                message=self._load_error or "Kokoro TTS service is unavailable.",
                status_code=503,
                recoverable=True,
                request_id=req_id,
            )

        clean_text, voice, speed = self.validate_request(text, voice_id, rate)
        start_time = time.perf_counter()

        logger.info(f"[KokoroTTS:{req_id}] Synthesizing: '{clean_text[:40]}...' (voice={voice}, rate={speed})")

        try:
            samples, sample_rate = self._kokoro_instance.create(
                clean_text,
                voice=voice,
                speed=speed,
                lang="en-us",
            )

            buf = io.BytesIO()
            sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
            audio_bytes = buf.getvalue()

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            metadata = {
                "request_id": req_id,
                "provider": "kokoro_onnx",
                "voice": voice,
                "speed": speed,
                "sample_rate": sample_rate,
                "latency_ms": latency_ms,
                "audio_length_bytes": len(audio_bytes),
                "media_type": "audio/wav",
            }

            return audio_bytes, "audio/wav", metadata

        except KokoroTTSError:
            raise
        except Exception as e:
            logger.error(f"[KokoroTTS:{req_id}] Synthesis error: {e}")
            raise KokoroTTSError(
                message=f"Kokoro synthesis failed: {str(e)}",
                status_code=500,
                recoverable=True,
                request_id=req_id,
            )


# Global Singleton Instance
_kokoro_service: Optional[KokoroTTSService] = None


def get_kokoro_tts_service() -> KokoroTTSService:
    """Retrieve or create the global Kokoro TTS service singleton."""
    global _kokoro_service
    if _kokoro_service is None:
        _kokoro_service = KokoroTTSService()
    return _kokoro_service
