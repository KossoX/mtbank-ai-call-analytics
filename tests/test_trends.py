from collections.abc import Iterable
from typing import Any

import pytest
from openai.types.chat import ChatCompletionMessageParam

from agents.trends import TrendsAgent


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(
        self,
        messages: Iterable[ChatCompletionMessageParam],
    ) -> str:
        return self.response


def _call(
    topic: str,
    score: int,
    passed: bool,
) -> dict[str, Any]:
    return {
        "analysis": {
            "classification": {
                "topic": topic,
                "priority": "medium",
            },
            "quality_score": {"total": score},
            "compliance": {"passed": passed},
            "summary": f"Звонок по теме {topic}.",
            "action_items": ["Проверить обращение."],
        }
    }


def test_trends_returns_statistics_and_llm_patterns() -> None:
    agent = TrendsAgent(
        llm=FakeLLM(
            '{"patterns": ["Повторяются вопросы по картам."], '
            '"recommendations": ["Обновить FAQ по картам."]}'
        )
    )

    result = agent.analyze(
        [
            _call("карты", 75, True),
            _call("карты", 50, False),
            _call("кредиты", 100, True),
        ]
    )

    assert result["total_calls"] == 3
    assert result["topic_distribution"] == {
        "карты": 2,
        "кредиты": 1,
    }
    assert result["average_quality_score"] == 75.0
    assert result["compliance_failure_rate"] == pytest.approx(
        1 / 3,
        abs=0.0001,
    )
    assert result["patterns"] == ["Повторяются вопросы по картам."]


def test_trends_requires_two_calls() -> None:
    agent = TrendsAgent(llm=FakeLLM('{"patterns": ["p"], "recommendations": ["r"]}'))

    with pytest.raises(ValueError, match="at least two"):
        agent.analyze([_call("карты", 75, True)])


def test_trends_rejects_invalid_llm_list() -> None:
    agent = TrendsAgent(llm=FakeLLM('{"patterns": [], "recommendations": ["r"]}'))

    with pytest.raises(ValueError, match="patterns"):
        agent.analyze(
            [
                _call("карты", 75, True),
                _call("кредиты", 50, True),
            ]
        )
