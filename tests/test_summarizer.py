from collections.abc import Iterable

import pytest
from openai.types.chat import ChatCompletionMessageParam

from agents.summarizer import SummarizerAgent


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(
        self,
        messages: Iterable[ChatCompletionMessageParam],
    ) -> str:
        return self.response


def test_summarizer_returns_summary_and_action_items() -> None:
    agent = SummarizerAgent(
        llm=FakeLLM(
            """
            {
              "summary": "Клиент обратился по вопросу кредита. Оператор объяснил способ подачи заявки. Клиент согласился подать заявку онлайн.",
              "action_items": [
                "Отправить клиенту инструкцию на email"
              ]
            }
            """
        )
    )

    result = agent.analyze("Тестовый диалог.")

    assert result == {
        "summary": (
            "Клиент обратился по вопросу кредита. "
            "Оператор объяснил способ подачи заявки. "
            "Клиент согласился подать заявку онлайн."
        ),
        "action_items": [
            "Отправить клиенту инструкцию на email",
        ],
    }


def test_summarizer_accepts_empty_action_items() -> None:
    agent = SummarizerAgent(
        llm=FakeLLM(
            """
            {
              "summary": "Клиент получил необходимую информацию. Дополнительные действия не требуются. Разговор завершён.",
              "action_items": []
            }
            """
        )
    )

    result = agent.analyze("Тестовый диалог.")

    assert result["action_items"] == []


def test_summarizer_rejects_invalid_json() -> None:
    agent = SummarizerAgent(llm=FakeLLM("это не json"))

    with pytest.raises(ValueError, match="invalid JSON"):
        agent.analyze("Тестовый диалог.")


def test_summarizer_rejects_non_list_action_items() -> None:
    agent = SummarizerAgent(
        llm=FakeLLM(
            '{"summary": "Тестовое резюме.", "action_items": "нет"}'
        )
    )

    with pytest.raises(ValueError, match="action_items list"):
        agent.analyze("Тестовый диалог.")


def test_summarizer_rejects_empty_transcript() -> None:
    agent = SummarizerAgent(
        llm=FakeLLM('{"summary": "Текст.", "action_items": []}')
    )

    with pytest.raises(ValueError, match="must not be empty"):
        agent.analyze("")