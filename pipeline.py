"""
title: MTB Bank Call Analytics
author: Kasyanchyk Artsiom
version: 1.1.0
description: Анализ звонков контакт-центра МТБанка.
requirements: httpx
"""

import json
import mimetypes
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field


UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".aiff",
    ".aif",
    ".flac",
}

EXTENSION_PRIORITY = {
    ".wav": 0,
    ".flac": 1,
    ".mp3": 2,
    ".m4a": 3,
    ".ogg": 4,
    ".aiff": 5,
    ".aif": 6,
}


class Pipeline:
    class Valves(BaseModel):
        API_URL: str = Field(
            default="http://api:8000/analyze",
            description="Адрес API анализа звонков.",
        )
        UPLOAD_DIRECTORY: str = Field(
            default="/app/backend/data/uploads",
            description="Каталог файлов Open WebUI.",
        )
        REQUEST_TIMEOUT_SECONDS: int = Field(
            default=300,
            description="Максимальное время анализа аудио.",
        )

    def __init__(self) -> None:
        self.name = "МТБанк — анализ звонков"
        self.valves = self.Valves()

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
    ) -> str:
        try:
            audio_path = self._find_audio_file(body)
        except ValueError as error:
            return f"### Не удалось начать анализ\n\n{error}"

        filename = audio_path.name
        content_type = (
            mimetypes.guess_type(filename)[0]
            or "application/octet-stream"
        )

        try:
            with audio_path.open("rb") as audio_file:
                response = httpx.post(
                    self.valves.API_URL,
                    files={
                        "file": (
                            filename,
                            audio_file,
                            content_type,
                        )
                    },
                    timeout=self.valves.REQUEST_TIMEOUT_SECONDS,
                )
        except httpx.TimeoutException:
            return (
                "### Анализ не завершён\n\n"
                "Сервис не успел обработать аудиозапись. "
                "Попробуйте ещё раз."
            )
        except httpx.RequestError as error:
            return (
                "### Сервис анализа недоступен\n\n"
                f"Не удалось обратиться к API: {error}"
            )

        try:
            payload = response.json()
        except ValueError:
            return (
                "### Некорректный ответ API\n\n"
                f"HTTP-статус: {response.status_code}. "
                "Ответ не является корректным JSON."
            )

        if not isinstance(payload, dict):
            return (
                "### Некорректный ответ API\n\n"
                "Ожидался JSON-объект."
            )

        if not response.is_success:
            detail = payload.get(
                "detail",
                "Неизвестная ошибка API.",
            )

            return (
                "### Анализ завершился ошибкой\n\n"
                f"HTTP-статус: {response.status_code}\n\n"
                f"{detail}"
            )

        return self._format_result(payload)

    def _find_audio_file(
        self,
        body: dict[str, Any],
    ) -> Path:
        upload_directory = Path(
            self.valves.UPLOAD_DIRECTORY
        ).resolve()

        if not upload_directory.is_dir():
            raise ValueError(
                "Каталог загруженных файлов Open WebUI недоступен."
            )

        candidates: list[Path] = []

        for reference in self._extract_file_references(body):
            file_id = reference.get("id")
            filename = reference.get("filename")
            explicit_path = reference.get("path")

            if explicit_path:
                candidates.append(Path(explicit_path))

            if file_id:
                candidates.extend(
                    upload_directory.glob(f"{file_id}_*")
                )

            if filename:
                safe_filename = Path(filename).name
                candidates.append(
                    upload_directory / safe_filename
                )
                candidates.extend(
                    upload_directory.glob(f"*_{safe_filename}")
                )

        audio_candidates = self._prepare_audio_candidates(
            candidates,
            upload_directory,
        )

        if not audio_candidates:
            raise ValueError(
                "Open WebUI сообщил о загруженном файле, "
                "но его идентификатор не удалось сопоставить "
                "с аудиофайлом в хранилище."
            )

        return audio_candidates[0]

    @staticmethod
    def _extract_file_references(
        value: Any,
    ) -> Iterator[dict[str, str]]:
        if isinstance(value, dict):
            reference: dict[str, str] = {}

            file_id = value.get("id")
            filename = (
                value.get("filename")
                or value.get("name")
            )
            explicit_path = value.get("path")

            if isinstance(file_id, str) and file_id:
                reference["id"] = file_id

            if isinstance(filename, str) and filename:
                reference["filename"] = filename

            if isinstance(explicit_path, str) and explicit_path:
                reference["path"] = explicit_path

            if reference:
                yield reference

            for nested_value in value.values():
                yield from Pipeline._extract_file_references(
                    nested_value
                )

            return

        if isinstance(value, list):
            for item in value:
                yield from Pipeline._extract_file_references(item)

            return

        if isinstance(value, str):
            uuid_match = UUID_PATTERN.search(value)

            if uuid_match:
                yield {"id": uuid_match.group(0)}

    @staticmethod
    def _prepare_audio_candidates(
        candidates: list[Path],
        upload_directory: Path,
    ) -> list[Path]:
        unique_candidates: dict[str, Path] = {}

        for candidate in candidates:
            try:
                resolved_candidate = candidate.resolve()
                resolved_candidate.relative_to(upload_directory)
            except (OSError, ValueError):
                continue

            extension = resolved_candidate.suffix.lower()

            if extension not in SUPPORTED_AUDIO_EXTENSIONS:
                continue

            if not resolved_candidate.is_file():
                continue

            unique_candidates[str(resolved_candidate)] = (
                resolved_candidate
            )

        return sorted(
            unique_candidates.values(),
            key=lambda path: (
                EXTENSION_PRIORITY.get(
                    path.suffix.lower(),
                    100,
                ),
                path.name,
            ),
        )

    @staticmethod
    def _format_result(payload: dict[str, Any]) -> str:
        transcript = str(payload.get("transcript", "")).strip()

        segments_value = payload.get("segments", [])
        segments = (
            segments_value
            if isinstance(segments_value, list)
            else []
        )

        analysis_value = payload.get("analysis", {})
        analysis = (
            analysis_value
            if isinstance(analysis_value, dict)
            else {}
        )

        classification_value = analysis.get(
            "classification",
            {},
        )
        classification = (
            classification_value
            if isinstance(classification_value, dict)
            else {}
        )

        quality_value = analysis.get("quality_score", {})
        quality = (
            quality_value
            if isinstance(quality_value, dict)
            else {}
        )

        checklist_value = quality.get("checklist", {})
        checklist = (
            checklist_value
            if isinstance(checklist_value, dict)
            else {}
        )

        compliance_value = analysis.get("compliance", {})
        compliance = (
            compliance_value
            if isinstance(compliance_value, dict)
            else {}
        )

        topic = classification.get("topic", "не определено")
        priority = classification.get(
            "priority",
            "не определён",
        )
        total = quality.get("total", 0)

        compliance_passed = bool(
            compliance.get("passed", False)
        )
        compliance_status = (
            "Пройдена"
            if compliance_passed
            else "Обнаружены нарушения"
        )

        lines = [
            "# Анализ звонка МТБанка",
            "",
            "## Основные показатели",
            "",
            f"- Тема обращения: **{topic}**",
            f"- Приоритет: **{priority}**",
            f"- Оценка качества: **{total}/100**",
            f"- Комплаенс-проверка: **{compliance_status}**",
            "",
            "## Чек-лист качества",
            "",
            Pipeline._checklist_line(
                "Приветствие",
                bool(checklist.get("greeting", False)),
            ),
            Pipeline._checklist_line(
                "Выявление потребности",
                bool(checklist.get("need_detection", False)),
            ),
            Pipeline._checklist_line(
                "Предложение решения",
                bool(checklist.get("solution_provided", False)),
            ),
            Pipeline._checklist_line(
                "Завершение разговора",
                bool(checklist.get("farewell", False)),
            ),
            "",
            "## Краткое содержание",
            "",
            str(
                analysis.get(
                    "summary",
                    "Краткое содержание отсутствует.",
                )
            ),
            "",
            "## Необходимые действия",
            "",
        ]

        action_items = analysis.get("action_items", [])

        if isinstance(action_items, list) and action_items:
            for action in action_items:
                lines.append(f"- {action}")
        else:
            lines.append("- Дополнительные действия не требуются.")

        lines.extend(
            [
                "",
                "## Комплаенс",
                "",
            ]
        )

        issues = compliance.get("issues", [])

        if isinstance(issues, list) and issues:
            for issue in issues:
                lines.append(f"- {issue}")
        else:
            lines.append("- Нарушения не обнаружены.")

        lines.extend(
            [
                "",
                "## Диалог по сегментам",
                "",
                "| Время | Спикер | Текст |",
                "|---|---|---|",
            ]
        )

        if segments:
            for segment in segments:
                if not isinstance(segment, dict):
                    continue

                start = Pipeline._format_timestamp(
                    segment.get("start", 0)
                )
                end = Pipeline._format_timestamp(
                    segment.get("end", 0)
                )
                speaker = Pipeline._escape_table_text(
                    segment.get("speaker", "Неизвестно")
                )
                text = Pipeline._escape_table_text(
                    segment.get("text", "")
                )

                lines.append(
                    f"| {start}–{end} | {speaker} | {text} |"
                )
        else:
            lines.append("| — | — | Сегменты отсутствуют |")

        lines.extend(
            [
                "",
                "## Полный транскрипт",
                "",
                transcript or "Транскрипт отсутствует.",
                "",
                "## Технический результат",
                "",
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def _checklist_line(
        label: str,
        completed: bool,
    ) -> str:
        marker = "✅" if completed else "❌"
        return f"- {marker} {label}"

    @staticmethod
    def _format_timestamp(value: Any) -> str:
        try:
            total_seconds = max(0, int(float(value)))
        except (TypeError, ValueError):
            total_seconds = 0

        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _escape_table_text(value: Any) -> str:
        return (
            str(value)
            .replace("|", "\\|")
            .replace("\n", " ")
            .strip()
        )