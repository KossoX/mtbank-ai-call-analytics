from pathlib import Path
from time import perf_counter
from typing import Any

from app.metrics import CALL_ERRORS_TOTAL, record_call_analysis
from app.orchestrator import AnalysisOrchestrator
from asr.diarizer import Diarizer
from asr.transcriber import Transcriber


class AudioAnalysisPipeline:
    def __init__(
        self,
        transcriber: Transcriber | None = None,
        diarizer: Diarizer | None = None,
        orchestrator: AnalysisOrchestrator | None = None,
    ) -> None:
        self._transcriber = transcriber or Transcriber()
        self._diarizer = diarizer or Diarizer()
        self._orchestrator = orchestrator or AnalysisOrchestrator()

    def analyze(self, audio_path: str | Path) -> dict[str, Any]:
        started_at = perf_counter()

        try:
            segments = self._transcriber.transcribe(audio_path)

            if not segments:
                raise ValueError(
                    "Audio transcription returned no segments."
                )

            diarized_segments = self._diarizer.assign_speakers(
                segments
            )
            transcript = self._build_transcript(
                diarized_segments
            )
            analysis = self._orchestrator.analyze(transcript)

            record_call_analysis(
                analysis,
                perf_counter() - started_at,
            )

            return {
                "transcript": transcript,
                "segments": diarized_segments,
                "analysis": analysis,
            }
        except Exception as error:
            CALL_ERRORS_TOTAL.labels(
                error_type=type(error).__name__
            ).inc()
            raise

    @staticmethod
    def _build_transcript(segments: list[dict[str, Any]]) -> str:
        lines: list[str] = []

        for segment in segments:
            speaker = str(segment["speaker"]).strip()
            text = str(segment["text"]).strip()

            if text:
                lines.append(f"{speaker}: {text}")

        if not lines:
            raise ValueError("Audio transcription contains no text.")

        return "\n".join(lines)
