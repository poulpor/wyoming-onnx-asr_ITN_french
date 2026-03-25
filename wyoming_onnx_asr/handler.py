"""Event handler for clients of the server."""

import asyncio
import logging
import os
import tempfile
import wave
from typing import Optional, Dict

import numpy as np
import soundfile as sf
from onnx_asr.adapters import AsrAdapter
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

# === NOUVEAU : ITN ===
from nemo_text_processing.inverse_text_normalization import InverseNormalizer

_LOGGER = logging.getLogger(__name__)


class NemoAsrEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        models: dict[str, AsrAdapter],
        model_lock: asyncio.Lock,
        *args,
        initial_prompt: Optional[str] = None,
        itn_cache_dir: str = "/app/cache",  # NOUVEAU : cache ITN persistant
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()
        self.models = models
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt
        self.request_language: Optional[str] = None
        
        # NOUVEAU : ITN multilingue (lazy init)
        self.itn_normalizers: Dict[str, InverseNormalizer] = {}
        self.itn_cache_dir = itn_cache_dir
        os.makedirs(self.itn_cache_dir, exist_ok=True)
        
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None

    def _get_itn(self, lang: str) -> InverseNormalizer:
        """Lazy init ITN pour une langue"""
        lang_code = self._lang_to_itn_code(lang)
        if lang_code not in self.itn_normalizers:
            try:
                self.itn_normalizers[lang_code] = InverseNormalizer(
                    lang=lang_code, cache_dir=self.itn_cache_dir
                )
                _LOGGER.info(f"ITN initialisé pour {lang_code}")
            except Exception as e:
                _LOGGER.warning(f"ITN {lang_code} échoué: {e}, fallback sans ITN")
                self.itn_normalizers[lang_code] = None
        return self.itn_normalizers[lang_code]

    def _lang_to_itn_code(self, lang: str) -> str:
        """Map lang vers code ITN"""
        mapping = {
            "fr": "fr",
            "en": "en", 
            "es": "es",
            "de": "de",
        }
        return mapping.get(lang.lower(), "en")  # fallback anglais

    def _apply_itn(self, text: str, lang: str) -> str:
        """Applique ITN + post-processing français pour format compact"""
        itn = self._get_itn(lang)
        if itn is None:
            return text
    
        try:
            normalized = itn.normalize(text)
        except Exception as e:
            _LOGGER.warning(f"ITN {lang} échoué: {e}")
            return text
    
        # NOUVEAU : Post-processing pour format français compact (12h30)
        if lang.lower() == "fr":
            normalized = self._french_time_compact(normalized)
    
        return normalized

    def _french_time_compact(self, text: str) -> str:
        """TOUTES les heures françaises : 9 h → 9h00 | 12 h → 12h00 | 23 h → 23h00"""
        import re
    
        # PRIORITÉ 1 : Heures + minutes (12 h 30 → 12h30)
        pattern_full = r'(\d{1,2})\s+h\s+(\d{1,2})'
        def replace_full(match):
            minutes = match.group(2).zfill(2)  # 5 → 05
            return f"{match.group(1)}h{minutes}"
    
        result = re.sub(pattern_full, replace_full, text)
    
        # PRIORITÉ 2 : TOUTES les heures seules → h00 (9 h → 9h00 | 14 h → 14h00)
        pattern_hours_all = r'(\d{1,2})\s+h\b'
        result = re.sub(pattern_hours_all, r'\1h00', result)
    
        # PRIORITÉ 3 : Cas h. (12 h. 30 → 12h30)
        pattern_dot = r'(\d{1,2})\s+h\.\s+(\d{1,2})'
        def replace_dot(match):
            minutes = match.group(2).zfill(2)
            return f"{match.group(1)}h{minutes}"
    
        result = re.sub(pattern_dot, replace_dot, result)
    
        return result
    
    
    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stopped. Transcribing with initial prompt=%s",
                self.initial_prompt,
            )
            assert self._wav_file is not None

            self._wav_file.close()
            self._wav_file = None

            waveform, sample_rate = sf.read(self._wav_path, dtype="float32")
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)

            # === MODIFIÉ : Langue + modèle ===
            lang = self.request_language or "en"
            model = None

            _LOGGER.info(f"Language requested: {lang}")
            _LOGGER.info(f"Available models: {list(self.models.keys())}")

            if lang == "en" and "en" in self.models:
                model = self.models["en"]
            elif "multi" in self.models:
                model = self.models["multi"]
            elif "en" in self.models:
                model = self.models["en"]

            if model is None:
                error_msg = f"Language '{self.request_language}' not supported. Available: {list(self.models.keys())}"
                await self.write_event(Transcript(text=f"ERROR: {error_msg}").event())
                return False

            async with self.model_lock:
                try:
                    _LOGGER.info(f"Starting transcription with model for '{lang}'")
                    raw_text = model.recognize(
                        waveform, language=lang, sample_rate=sample_rate
                    )
                    _LOGGER.info(f"ASR raw: {raw_text}")
                except Exception as e:
                    _LOGGER.error("ASR failed: %s", str(e))
                    await self.write_event(Transcript(text=f"ERROR: {str(e)}").event())
                    return False

            # === NOUVEAU : ITN après ASR ===
            final_text = self._apply_itn(raw_text, lang)
            _LOGGER.info(f"ITN {lang}: {final_text}")

            await self.write_event(Transcript(text=final_text).event())
            _LOGGER.debug("Completed request")

            self.request_language = None
            return False

        # ... reste identique (Transcribe, Describe)
        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self.request_language = transcribe.language
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
