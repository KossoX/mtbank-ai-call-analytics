import json
from typing import Any

from app.llm import LLMClient, LLMProvider

ALLOWED_TOPICS = {
    "кредиты",
    "карты",
    "переводы",
    "жалобы",
    "не определено",
}
ALLOWED_PRIORITIES = {"low", "medium", "high"}


class ClassifierAgent:
    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or LLMClient()

    def analyze(self, transcript: str) -> dict[str, str]:
        if not transcript.strip():
            raise ValueError("Transcript must not be empty.")

        response = self._llm.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Ты классификатор обращений контакт-центра МТБанка. "
                        "Определи тему обращения и приоритет. "
                        "Если клиент ещё не сообщил причину обращения "
                        "или информации недостаточно, используй тему "
                        '"не определено". '
                        "Верни только JSON без Markdown и дополнительных пояснений. "
                        'Формат: {"topic": "...", "priority": "..."}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Допустимые темы: кредиты, карты, переводы, жалобы, "
                        "не определено.\n"
                        "Допустимые приоритеты: low, medium, high.\n\n"
                        f"Транскрипт:\n{transcript}"
                    ),
                },
            ]
        )

        result = self._parse_response(response)

        topic = result.get("topic")
        priority = result.get("priority")

        if topic not in ALLOWED_TOPICS:
            raise ValueError(f"Unsupported topic returned by LLM: {topic}")

        if priority not in ALLOWED_PRIORITIES:
            raise ValueError(f"Unsupported priority returned by LLM: {priority}")

        return {"topic": topic, "priority": priority}

    @staticmethod
    def _parse_response(response: str) -> dict[str, Any]:
        cleaned = response.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise ValueError("Classifier returned invalid JSON.") from error

        if not isinstance(result, dict):
            raise ValueError("Classifier response must be a JSON object.")

        return result
