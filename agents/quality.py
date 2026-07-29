import json
from typing import Any

from app.llm import LLMClient, LLMProvider

CHECKLIST_FIELDS = (
    "greeting",
    "need_detection",
    "solution_provided",
    "farewell",
)


class QualityAgent:
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
                        "Ты агент контроля качества контакт-центра МТБанка. "
                        "Оцени действия оператора только по содержанию транскрипта. "
                        "Верни только JSON без Markdown и пояснений."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Проверь четыре критерия:\n"
                        "greeting — оператор поздоровался и представился;\n"
                        "need_detection — оператор выяснил потребность клиента;\n"
                        "solution_provided — оператор предложил решение или следующий шаг;\n"
                        "farewell — оператор корректно завершил разговор.\n\n"
                        "Верни JSON строго в формате:\n"
                        '{"checklist": {'
                        '"greeting": true, '
                        '"need_detection": true, '
                        '"solution_provided": true, '
                        '"farewell": true'
                        "}}\n\n"
                        f"Транскрипт:\n{transcript}"
                    ),
                },
            ]
        )

        result = self._parse_response(response)
        checklist = result.get("checklist")

        if not isinstance(checklist, dict):
            raise ValueError("Quality response must contain a checklist object.")

        validated_checklist: dict[str, bool] = {}

        for field in CHECKLIST_FIELDS:
            value = checklist.get(field)

            if not isinstance(value, bool):
                raise ValueError(f"Quality checklist field must be boolean: {field}")

            validated_checklist[field] = value

        total = sum(25 for value in validated_checklist.values() if value)

        return {
            "total": total,
            "checklist": validated_checklist,
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
            raise ValueError("Quality agent returned invalid JSON.") from error

        if not isinstance(result, dict):
            raise ValueError("Quality response must be a JSON object.")

        return result
