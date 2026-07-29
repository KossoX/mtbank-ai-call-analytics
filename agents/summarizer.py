import json
from typing import Any

from app.llm import LLMClient, LLMProvider


class SummarizerAgent:
    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or LLMClient()

    def analyze(self, transcript: str) -> dict[str, Any]:
        if not transcript.strip():
            raise ValueError("Transcript must not be empty.")

        response = self._llm.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Ты агент суммаризации звонков контакт-центра МТБанка. "
                        "Составь краткое резюме на русском языке. "
                        "Используй только факты из транскрипта. "
                        "Не придумывай решения, суммы, обещания или действия. "
                        "Верни только JSON без Markdown и пояснений."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Требования:\n"
                        "- summary: 3–5 предложений;\n"
                        "- action_items: список конкретных следующих действий;\n"
                        "- если действий не было, верни пустой список.\n\n"
                        "Верни JSON строго в формате:\n"
                        '{"summary": "текст из 3–5 предложений", '
                        '"action_items": ["действие 1"]}\n\n'
                        f"Транскрипт:\n{transcript}"
                    ),
                },
            ]
        )

        result = self._parse_response(response)

        summary = result.get("summary")
        action_items = result.get("action_items")

        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("Summary must be a non-empty string.")

        if not isinstance(action_items, list):
            raise ValueError("Summary response must contain an action_items list.")

        validated_action_items: list[str] = []

        for item in action_items:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Every action item must be a non-empty string.")

            validated_action_items.append(item.strip())

        return {
            "summary": summary.strip(),
            "action_items": validated_action_items,
        }

    @staticmethod
    def _parse_response(response: str) -> dict[str, Any]:
        cleaned = response.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise ValueError("Summarizer returned invalid JSON.") from error

        if not isinstance(result, dict):
            raise ValueError("Summary response must be a JSON object.")

        return result
