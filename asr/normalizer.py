import re

BANK_NAME_PATTERNS = (
    r"\bmk\s*bank\b",
    r"\bmt\s*bank\b",
    r"\bмк\s*банк\b",
    r"\bмт\s*банк\b",
)

SENTENCE_START_FIXES = (
    (re.compile(r"([.!?…])([А-ЯЁA-Z])"), r"\1 \2"),
    (re.compile(r"([,;:])([^\s])"), r"\1 \2"),
    (re.compile(r"(?<!\s)([«\"“])"), r" \1"),
)


def _restore_basic_spacing(text: str) -> str:
    restored = re.sub(r"\s+", " ", text)
    restored = re.sub(r"\s+([,.;:!?])", r"\1", restored)

    for pattern, replacement in SENTENCE_START_FIXES:
        restored = pattern.sub(replacement, restored)

    restored = re.sub(r"\bЯ(?=[а-яё])", "Я ", restored)
    restored = re.sub(r"\s+", " ", restored)

    return restored.strip()


def normalize_transcript_text(text: str) -> str:
    normalized = _restore_basic_spacing(text)

    for pattern in BANK_NAME_PATTERNS:
        normalized = re.sub(
            pattern,
            "МТБанк",
            normalized,
            flags=re.IGNORECASE,
        )

    return normalized
