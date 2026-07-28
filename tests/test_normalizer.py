from asr.normalizer import normalize_transcript_text


def test_normalizer_replaces_latin_bank_name_variants() -> None:
    text = "Добрый день, MkBank. Я звоню в MT Bank."

    result = normalize_transcript_text(text)

    assert result == "Добрый день, МТБанк. Я звоню в МТБанк."


def test_normalizer_replaces_cyrillic_bank_name_variants() -> None:
    text = "Здравствуйте, МК Банк и МТ Банк."

    result = normalize_transcript_text(text)

    assert result == "Здравствуйте, МТБанк и МТБанк."


def test_normalizer_preserves_unrelated_text() -> None:
    text = "Клиент хочет узнать условия кредита."

    result = normalize_transcript_text(text)

    assert result == text


def test_normalizer_strips_outer_whitespace() -> None:
    result = normalize_transcript_text("  Добрый день.  ")

    assert result == "Добрый день."