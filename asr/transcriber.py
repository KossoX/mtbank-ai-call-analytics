from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from app.config import get_settings


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Transcriber:
    def __init__(self) -> None:
        settings = get_settings()

        self._model = WhisperModel(
            settings.whisper_model,
            device="cpu",
            compute_type="int8",
        )

    def transcribe(self, audio_path: str | Path) -> list[dict[str, Any]]:
        path = Path(audio_path)

        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        segments, _info = self._model.transcribe(
            str(path),
            beam_size=5,
            vad_filter=True,
        )

        result: list[dict[str, Any]] = []

        for segment in segments:
            transcript_segment = TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
            )
            result.append(transcript_segment.to_dict())

        return result