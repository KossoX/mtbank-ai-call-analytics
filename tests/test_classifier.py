from collections.abc import Iterable

import pytest
from openai.types.chat import ChatCompletionMessageParam

from agents.classifier import ClassifierAgent


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(
        self,
        messages: Iterable[ChatCompletionMessageParam],
    ) -> str:
        return self.response


def test_classifier_returns_valid_result() -> None:
    agent = ClassifierAgent(llm=FakeLLM('{"topic": "кредиты", "priority": "medium"}'))

    result = agent.analyze("Клиент спрашивает условия кредита.")

    assert result == {
        "topic": "кредиты",
        "priority": "medium",
    }


def test_classifier_rejects_invalid_json() -> None:
    agent = ClassifierAgent(llm=FakeLLM("это не json"))

    with pytest.raises(ValueError, match="invalid JSON"):
        agent.analyze("Клиент спрашивает условия кредита.")


def test_classifier_rejects_unknown_topic() -> None:
    agent = ClassifierAgent(llm=FakeLLM('{"topic": "страхование", "priority": "low"}'))

    with pytest.raises(ValueError, match="Unsupported topic"):
        agent.analyze("Клиент спрашивает про страхование.")


def test_classifier_rejects_empty_transcript() -> None:
    agent = ClassifierAgent(llm=FakeLLM("{}"))

    with pytest.raises(ValueError, match="must not be empty"):
        agent.analyze("")


def test_classifier_returns_undefined_topic_when_reason_is_missing() -> None:
    agent = ClassifierAgent(
        llm=FakeLLM('{"topic": "не определено", "priority": "low"}')
    )

    result = agent.analyze(
        "Оператор: Добрый день, МТБанк, меня зовут Анна. Чем могу помочь?"
    )

    assert result == {
        "topic": "не определено",
        "priority": "low",
    }
