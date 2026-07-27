import json
from typing import Any

from app.llm import LLMClient, LLMProvider


class ComplianceAgent:
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
                        "Ты compliance-агент контакт-центра МТБанка. "
                        "Проверяй только высказывания оператора. "
                        "Не приписывай оператору слова клиента. "
                        "Верни только JSON без Markdown и пояснений."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Найди следующие нарушения:\n"
                        "1. Гарантия одобрения кредита или другого продукта.\n"
                        "2. Гарантия конкретной ставки без оговорок об условиях.\n"
                        "3. Просьба сообщить пароль, CVV/CVC или SMS-код.\n"
                        "4. Утверждение, что добровольная страховка обязательна.\n"
                        "5. Некорректные обещания от имени банка.\n"
                        "6. Для кредитного предложения — отсутствие пояснения, "
                        "что окончательные условия зависят от рассмотрения заявки.\n\n"
                        "Верни JSON строго в формате:\n"
                        '{"issues": ["описание нарушения"]}\n'
                        "Если нарушений нет, верни:\n"
                        '{"issues": []}\n\n'
                        f"Транскрипт:\n{transcript}"
                    ),
                },
            ]
        )

        result = self._parse_response(response)
        issues = result.get("issues")

        if not isinstance(issues, list):
            raise ValueError("Compliance response must contain an issues list.")

        validated_issues: list[str] = []

        for issue in issues:
            if not isinstance(issue, str) or not issue.strip():
                raise ValueError(
                    "Every compliance issue must be a non-empty string."
                )

            validated_issues.append(issue.strip())

        return {
            "passed": not validated_issues,
            "issues": validated_issues,
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
            raise ValueError("Compliance agent returned invalid JSON.") from error

        if not isinstance(result, dict):
            raise ValueError("Compliance response must be a JSON object.")

        return result