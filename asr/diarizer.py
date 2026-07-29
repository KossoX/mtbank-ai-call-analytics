import re
from typing import Any


SPEAKERS = ("Оператор", "Клиент")

OPERATOR_HINTS = (
    "добрый день",
    "чем могу помочь",
    "подскажите",
    "какая сумма",
    "перевод был",
    "заявку можно подать",
    "обычно нужен паспорт",
    "в приложении",
    "историю операции",
    "и последние четыре",
    "уточним",
    "проверю",
    "мы можем",
    "я оформлю",
    "я передам",
    "я рекомендую",
    "если появятся вопросы",
    "пожалуйста",
    "хорошего дня",
    "межбанковские переводы",
    "мешбанковские переводы",
    "отмена обычно невозможна",
)

CLIENT_HINTS = (
    r"\bя перев[её]л\b",
    r"\bя отправлял\b",
    r"\bя хотел\b",
    r"\bя проверил\b",
    r"\bмне\b",
    r"\bхочу\b",
    r"\bнужно\b",
    r"\bу меня\b",
    r"\bсумма\b",
    r"\bтам написано\b",
    r"\bможно ли\b",
    r"\bкакие документы\b",
    r"\bреквизиты\b",
    r"\bхорошо, тогда\b",
    r"\bпонял\b",
    r"\bпоняла\b",
    r"\bспасибо за\b",
)


class Diarizer:
    @staticmethod
    def _guess_speaker(text: str, fallback: str) -> str:
        normalized = text.strip().lower()

        if not normalized:
            return fallback

        if any(hint in normalized for hint in OPERATOR_HINTS):
            return "Оператор"

        if any(
            re.search(pattern, normalized)
            for pattern in CLIENT_HINTS
        ):
            return "Клиент"

        return fallback

    @staticmethod
    def _is_continuation(text: str) -> bool:
        normalized = text.strip()

        if not normalized:
            return False

        return normalized[-1] not in ".!?"

    def assign_speakers(
        self,
        segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        fallback = SPEAKERS[0]

        for segment in segments:
            enriched_segment = dict(segment)
            text = str(enriched_segment.get("text", ""))

            if result and self._is_continuation(
                str(result[-1].get("text", ""))
            ):
                guessed_speaker = str(result[-1]["speaker"])
            else:
                guessed_speaker = self._guess_speaker(
                    text,
                    fallback,
                )

            enriched_segment["speaker"] = guessed_speaker
            result.append(enriched_segment)

            if not self._is_continuation(text):
                fallback = (
                    "Клиент"
                    if guessed_speaker == "Оператор"
                    else "Оператор"
                )

        return result