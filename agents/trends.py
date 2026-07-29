import json
from collections import Counter
from typing import Any

from app.llm import LLMClient, LLMProvider


class TrendsAgent:
    """Finds recurring patterns across several completed call analyses."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or LLMClient()

    def analyze(
        self,
        calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if len(calls) < 2:
            raise ValueError(
                "Trend analysis requires at least two calls."
            )

        statistics = self._calculate_statistics(calls)
        compact_calls = [
            {
                "topic": self._nested(
                    call,
                    "analysis",
                    "classification",
                    "topic",
                ),
                "priority": self._nested(
                    call,
                    "analysis",
                    "classification",
                    "priority",
                ),
                "quality_score": self._nested(
                    call,
                    "analysis",
                    "quality_score",
                    "total",
                ),
                "compliance_passed": self._nested(
                    call,
                    "analysis",
                    "compliance",
                    "passed",
                ),
                "summary": self._nested(
                    call,
                    "analysis",
                    "summary",
                ),
                "action_items": self._nested(
                    call,
                    "analysis",
                    "action_items",
                ),
            }
            for call in calls
        ]

        response = self._llm.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Ты агент трендов контакт-центра МТБанка. "
                        "Найди повторяющиеся причины обращений, проблемы "
                        "качества, compliance-риски и возможности улучшения. "
                        "Опирайся только на входные данные. Верни только JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Верни JSON строго в формате:\n"
                        '{"patterns": ["..."], '
                        '"recommendations": ["..."]}\n'
                        "Каждый список должен содержать 1-5 конкретных "
                        "непустых пунктов.\n\n"
                        f"Статистика:\n{json.dumps(statistics, ensure_ascii=False)}"
                        "\n\nЗвонки:\n"
                        f"{json.dumps(compact_calls, ensure_ascii=False)}"
                    ),
                },
            ]
        )
        llm_result = self._parse_response(response)

        return {
            **statistics,
            "patterns": self._validate_string_list(
                llm_result.get("patterns"),
                "patterns",
            ),
            "recommendations": self._validate_string_list(
                llm_result.get("recommendations"),
                "recommendations",
            ),
        }

    @staticmethod
    def _nested(
        value: dict[str, Any],
        *keys: str,
    ) -> Any:
        current: Any = value

        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)

        return current

    @classmethod
    def _calculate_statistics(
        cls,
        calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        topics: Counter[str] = Counter()
        scores: list[float] = []
        compliance_failures = 0

        for call in calls:
            topic = cls._nested(
                call,
                "analysis",
                "classification",
                "topic",
            )
            topics[str(topic or "не определено")] += 1

            score = cls._nested(
                call,
                "analysis",
                "quality_score",
                "total",
            )
            if isinstance(score, (int, float)):
                scores.append(float(score))

            passed = cls._nested(
                call,
                "analysis",
                "compliance",
                "passed",
            )
            if passed is False:
                compliance_failures += 1

        average_quality = (
            round(sum(scores) / len(scores), 2)
            if scores
            else 0.0
        )

        return {
            "total_calls": len(calls),
            "topic_distribution": dict(
                sorted(
                    topics.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "average_quality_score": average_quality,
            "compliance_failure_rate": round(
                compliance_failures / len(calls),
                4,
            ),
        }

    @staticmethod
    def _validate_string_list(
        value: Any,
        field: str,
    ) -> list[str]:
        if not isinstance(value, list) or not 1 <= len(value) <= 5:
            raise ValueError(
                f"Trends response {field} must contain 1-5 items."
            )

        result: list[str] = []

        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"Every trends {field} item must be a non-empty string."
                )
            result.append(item.strip())

        return result

    @staticmethod
    def _parse_response(response: str) -> dict[str, Any]:
        cleaned = response.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Trends agent returned invalid JSON."
            ) from error

        if not isinstance(result, dict):
            raise ValueError(
                "Trends response must be a JSON object."
            )

        return result
