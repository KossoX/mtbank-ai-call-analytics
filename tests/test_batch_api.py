from typing import Any

from fastapi.testclient import TestClient

from app.api import create_app


class FakePipeline:
    def analyze(self, audio_path: str) -> dict[str, Any]:
        return {
            "transcript": "Оператор: Тест.",
            "segments": [],
            "analysis": {
                "classification": {
                    "topic": "карты",
                    "priority": "low",
                },
                "quality_score": {"total": 80},
                "compliance": {"passed": True, "issues": []},
                "summary": "Тестовый звонок.",
                "action_items": [],
            },
        }


class FakeTrends:
    def analyze(self, calls: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "total_calls": len(calls),
            "topic_distribution": {"карты": len(calls)},
            "average_quality_score": 80.0,
            "compliance_failure_rate": 0.0,
            "patterns": ["Повторяющаяся тема карт."],
            "recommendations": ["Обновить FAQ."],
        }


def test_batch_endpoint_analyzes_calls_and_trends() -> None:
    client = TestClient(
        create_app(
            pipeline=FakePipeline(),
            trends_agent=FakeTrends(),
        )
    )

    response = client.post(
        "/analyze-batch",
        files=[
            ("files", ("one.wav", b"audio-1", "audio/wav")),
            ("files", ("two.wav", b"audio-2", "audio/wav")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["calls"]) == 2
    assert payload["trends"]["total_calls"] == 2


def test_batch_endpoint_requires_two_files() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.post(
        "/analyze-batch",
        files=[("files", ("one.wav", b"audio-1", "audio/wav"))],
    )

    assert response.status_code == 400
