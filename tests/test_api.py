from typing import Any

from fastapi.testclient import TestClient

from app.api import create_app


class FakePipeline:
    def analyze(self, audio_path: str) -> dict[str, Any]:
        return {
            "transcript": "Оператор: Добрый день.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Добрый день.",
                    "speaker": "Оператор",
                }
            ],
            "analysis": {
                "classification": {
                    "topic": "не определено",
                    "priority": "low",
                }
            },
        }


def test_health_endpoint() -> None:
    client = TestClient(
        create_app(pipeline=FakePipeline())  # type: ignore[arg-type]
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_endpoint_accepts_audio_file() -> None:
    client = TestClient(
        create_app(pipeline=FakePipeline())  # type: ignore[arg-type]
    )

    response = client.post(
        "/analyze",
        files={
            "file": (
                "sample.wav",
                b"fake wav content",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["transcript"] == "Оператор: Добрый день."


def test_analyze_endpoint_rejects_empty_file() -> None:
    client = TestClient(
        create_app(pipeline=FakePipeline())  # type: ignore[arg-type]
    )

    response = client.post(
        "/analyze",
        files={
            "file": (
                "empty.wav",
                b"",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Uploaded audio file is empty."
    )