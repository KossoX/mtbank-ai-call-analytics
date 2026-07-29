from collections.abc import Iterable

import pytest
from openai.types.chat import ChatCompletionMessageParam

from agents.quality import QualityAgent


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(
        self,
        messages: Iterable[ChatCompletionMessageParam],
    ) -> str:
        return self.response


def test_quality_calculates_total_from_checklist() -> None:
    agent = QualityAgent(
        llm=FakeLLM(
            """
            {
              "checklist": {
                "greeting": true,
                "need_detection": true,
                "solution_provided": true,
                "farewell": false
              }
            }
            """
        )
    )

    result = agent.analyze("Тестовый диалог.")

    assert result == {
        "total": 75,
        "checklist": {
            "greeting": True,
            "need_detection": True,
            "solution_provided": True,
            "farewell": False,
        },
    }


def test_quality_rejects_invalid_json() -> None:
    agent = QualityAgent(llm=FakeLLM("это не json"))

    with pytest.raises(ValueError, match="invalid JSON"):
        agent.analyze("Тестовый диалог.")


def test_quality_rejects_non_boolean_checklist_value() -> None:
    agent = QualityAgent(
        llm=FakeLLM(
            """
            {
              "checklist": {
                "greeting": "yes",
                "need_detection": true,
                "solution_provided": true,
                "farewell": false
              }
            }
            """
        )
    )

    with pytest.raises(ValueError, match="must be boolean"):
        agent.analyze("Тестовый диалог.")


def test_quality_rejects_empty_transcript() -> None:
    agent = QualityAgent(llm=FakeLLM("{}"))

    with pytest.raises(ValueError, match="must not be empty"):
        agent.analyze("")
