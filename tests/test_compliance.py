from collections.abc import Iterable

import pytest
from openai.types.chat import ChatCompletionMessageParam

from agents.compliance import ComplianceAgent


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(
        self,
        messages: Iterable[ChatCompletionMessageParam],
    ) -> str:
        return self.response


def test_compliance_passes_when_issues_are_empty() -> None:
    agent = ComplianceAgent(llm=FakeLLM('{"issues": []}'))

    result = agent.analyze("Корректный тестовый диалог.")

    assert result == {
        "passed": True,
        "issues": [],
    }


def test_compliance_fails_when_issues_exist() -> None:
    agent = ComplianceAgent(
        llm=FakeLLM(
            """
            {
              "issues": [
                "Оператор гарантировал одобрение кредита.",
                "Оператор запросил SMS-код."
              ]
            }
            """
        )
    )

    result = agent.analyze("Диалог с нарушениями.")

    assert result == {
        "passed": False,
        "issues": [
            "Оператор гарантировал одобрение кредита.",
            "Оператор запросил SMS-код.",
        ],
    }


def test_compliance_rejects_invalid_json() -> None:
    agent = ComplianceAgent(llm=FakeLLM("это не json"))

    with pytest.raises(ValueError, match="invalid JSON"):
        agent.analyze("Тестовый диалог.")


def test_compliance_rejects_non_list_issues() -> None:
    agent = ComplianceAgent(llm=FakeLLM('{"issues": "нет нарушений"}'))

    with pytest.raises(ValueError, match="issues list"):
        agent.analyze("Тестовый диалог.")


def test_compliance_rejects_empty_transcript() -> None:
    agent = ComplianceAgent(llm=FakeLLM('{"issues": []}'))

    with pytest.raises(ValueError, match="must not be empty"):
        agent.analyze("")
