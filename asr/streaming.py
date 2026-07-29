import wave
from dataclasses import asdict, dataclass
from io import BytesIO
from threading import Lock
from typing import Any

from faster_whisper import WhisperModel

from app.config import get_settings
from asr.normalizer import normalize_transcript_text

SILENCE_RMS_THRESHOLD = 1500.0


@dataclass(frozen=True)
class StreamingSegment:
    start: float
    end: float
    raw_text: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StreamingTranscriber:
    """Low-latency ASR for independent PCM16 mono chunks."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.realtime_whisper_model
        self._model: WhisperModel | None = None
        self._model_lock = Lock()
        self._inference_lock = Lock()

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = WhisperModel(
                        self._model_name,
                        device="cpu",
                        compute_type="int8",
                    )

        return self._model

    def preload(self) -> None:
        """Load weights and warm the inference path before serving clients."""
        self._get_model()
        self.transcribe_pcm(
            b"\x00\x00" * 16000,
            sample_rate=16000,
            language="ru",
        )

    def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate: int,
        language: str = "ru",
    ) -> list[dict[str, Any]]:
        if not pcm_bytes:
            return []

        if len(pcm_bytes) % 2:
            raise ValueError("PCM16 payload must contain complete samples.")

        if sample_rate not in {8000, 16000, 24000, 48000}:
            raise ValueError("sample_rate must be 8000, 16000, 24000 or 48000.")

        if self._is_silence(pcm_bytes):
            return []

        wav_buffer = BytesIO()

        with wave.open(wav_buffer, "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(sample_rate)
            audio_file.writeframes(pcm_bytes)

        wav_buffer.seek(0)
        with self._inference_lock:
            segments, _info = self._get_model().transcribe(
                wav_buffer,
                language=language,
                beam_size=1,
                best_of=1,
                max_new_tokens=48,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
            )

        result: list[dict[str, Any]] = []

        for segment in segments:
            raw_text = segment.text.strip()

            if not raw_text:
                continue

            segment_start = float(segment.start)
            segment_end = float(segment.end)
            chunk_duration = len(pcm_bytes) / (sample_rate * 2)

            if segment_end <= segment_start:
                segment_end = chunk_duration

            result.append(
                StreamingSegment(
                    start=segment_start,
                    end=segment_end,
                    raw_text=raw_text,
                    text=normalize_transcript_text(raw_text),
                ).to_dict()
            )

        return result

    @staticmethod
    def _is_silence(pcm_bytes: bytes) -> bool:
        samples = memoryview(pcm_bytes).cast("h")

        if not samples:
            return True

        mean_square = sum(int(sample) * int(sample) for sample in samples) / len(
            samples
        )
        return mean_square**0.5 < SILENCE_RMS_THRESHOLD
