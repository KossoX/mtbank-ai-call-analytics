from asr.diarizer import Diarizer


def test_diarizer_assigns_speakers_in_alternating_order() -> None:
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Добрый день."},
        {"start": 2.0, "end": 4.0, "text": "Хочу узнать условия кредита."},
        {"start": 4.0, "end": 6.0, "text": "Какая сумма вас интересует?"},
    ]

    result = Diarizer().assign_speakers(segments)

    assert [segment["speaker"] for segment in result] == [
        "Оператор",
        "Клиент",
        "Оператор",
    ]


def test_diarizer_keeps_same_speaker_for_continued_sentence() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Заявку можно подать в мобильном приложении",
        },
        {
            "start": 2.0,
            "end": 4.0,
            "text": "или в любом отделении банка.",
        },
        {
            "start": 4.0,
            "end": 6.0,
            "text": "Какие документы понадобятся для оформления?",
        },
        {
            "start": 6.0,
            "end": 8.0,
            "text": "Обычно нужен паспорт, а окончательный список документов",
        },
        {
            "start": 8.0,
            "end": 10.0,
            "text": "зависит от результата рассмотрения заявки.",
        },
    ]

    result = Diarizer().assign_speakers(segments)

    assert [segment["speaker"] for segment in result] == [
        "Оператор",
        "Оператор",
        "Клиент",
        "Оператор",
        "Оператор",
    ]


def test_diarizer_recognizes_client_thanks() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Спасибо за подробную консультацию.",
        }
    ]

    result = Diarizer().assign_speakers(segments)

    assert result[0]["speaker"] == "Клиент"


def test_diarizer_preserves_original_segment_data() -> None:
    segments = [
        {"start": 1.5, "end": 3.25, "text": "Проверка сегмента."},
    ]

    result = Diarizer().assign_speakers(segments)

    assert result == [
        {
            "start": 1.5,
            "end": 3.25,
            "text": "Проверка сегмента.",
            "speaker": "Оператор",
        }
    ]


def test_diarizer_returns_empty_list_for_empty_input() -> None:
    assert Diarizer().assign_speakers([]) == []
