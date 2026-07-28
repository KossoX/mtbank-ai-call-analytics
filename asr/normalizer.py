import re


BANK_NAME_PATTERNS = (
    r"\bmk\s*bank\b",
    r"\bmt\s*bank\b",
    r"\bмк\s*банк\b",
    r"\bмт\s*банк\b",
)


def normalize_transcript_text(text: str) -> str:
    normalized = text.strip()

    for pattern in BANK_NAME_PATTERNS:
        normalized = re.sub(
            pattern,
            "МТБанк",
            normalized,
            flags=re.IGNORECASE,
        )

    return normalized