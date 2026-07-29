import pytest

from app.pipeline import AudioAnalysisPipeline


class FakeTranscriber:
    def transcribe(self, audio_path: str) -> list[dict[str, object]]:
        return [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Добрый день.",
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "Хочу узнать условия кредита.",
            },
        ]


class FakeDiarizer:
    def assign_speakers(
        self,
        segments: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return [
            {**segments[0], "speaker": "Оператор"},
            {**segments[1], "speaker": "Клиент"},
        ]


class FakeOrchestrator:
    def analyze(self, transcript: str) -> dict[str, object]:
        return {
            "received_transcript": transcript,
        }


def test_pipeline_builds_transcript_and_analysis() -> None:
    pipeline = AudioAnalysisPipeline(
        transcriber=FakeTranscriber(),  # type: ignore[arg-type]
        diarizer=FakeDiarizer(),  # type: ignore[arg-type]
        orchestrator=FakeOrchestrator(),  # type: ignore[arg-type]
    )

    result = pipeline.analyze("test.wav")

    assert result["transcript"] == (
        "Оператор: Добрый день.\nКлиент: Хочу узнать условия кредита."
    )
    assert result["analysis"] == {
        "received_transcript": (
            "Оператор: Добрый день.\nКлиент: Хочу узнать условия кредита."
        )
    }


def test_pipeline_rejects_empty_transcription() -> None:
    class EmptyTranscriber:
        def transcribe(self, audio_path: str) -> list[dict[str, object]]:
            return []

    pipeline = AudioAnalysisPipeline(
        transcriber=EmptyTranscriber(),  # type: ignore[arg-type]
        diarizer=FakeDiarizer(),  # type: ignore[arg-type]
        orchestrator=FakeOrchestrator(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="no segments"):
        pipeline.analyze("test.wav")
