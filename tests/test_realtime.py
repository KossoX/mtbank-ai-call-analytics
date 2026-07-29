from typing import Any

from fastapi.testclient import TestClient

from app.api import create_app


class FakePipeline:
    def analyze(self, audio_path: str) -> dict[str, Any]:
        return {"transcript": "", "segments": [], "analysis": {}}


class FakeStreamingTranscriber:
    def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate: int,
        language: str = "ru",
    ) -> list[dict[str, Any]]:
        assert sample_rate == 16000
        assert language == "ru"
        return [
            {
                "start": 0.0,
                "end": 0.5,
                "raw_text": "Тестовый фрагмент.",
                "text": "Тестовый фрагмент.",
            }
        ]


def test_realtime_websocket_returns_partial_and_completion() -> None:
    client = TestClient(
        create_app(
            pipeline=FakePipeline(),
            streaming_transcriber=FakeStreamingTranscriber(),
        )
    )

    with client.websocket_connect("/ws/transcribe") as websocket:
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_json(
            {
                "type": "start",
                "sample_rate": 16000,
                "language": "ru",
            }
        )
        assert websocket.receive_json()["type"] == "started"
        websocket.send_bytes(b"\x00\x00" * 16000 * 2)
        partial = websocket.receive_json()
        assert partial["type"] == "partial"
        assert partial["text"] == "Тестовый фрагмент."
        assert partial["latency_target_met"] is True
        assert websocket.receive_json()["type"] == "partial"
        websocket.send_json({"type": "stop"})
        assert websocket.receive_json()["type"] == "completed"


def test_realtime_websocket_rejects_invalid_control_json() -> None:
    client = TestClient(
        create_app(
            pipeline=FakePipeline(),
            streaming_transcriber=FakeStreamingTranscriber(),
        )
    )

    with client.websocket_connect("/ws/transcribe") as websocket:
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_text("not-json")
        assert websocket.receive_json()["type"] == "error"
