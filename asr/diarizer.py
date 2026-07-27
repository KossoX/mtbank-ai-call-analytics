from typing import Any


SPEAKERS = ("Оператор", "Клиент")


class Diarizer:
    def assign_speakers(
        self,
        segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for index, segment in enumerate(segments):
            enriched_segment = dict(segment)
            enriched_segment["speaker"] = SPEAKERS[index % len(SPEAKERS)]
            result.append(enriched_segment)

        return result