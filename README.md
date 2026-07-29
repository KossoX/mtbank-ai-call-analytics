# MTBank AI Call Analytics

Прототип речевой аналитики для контакт-центра МТБанка. Сервис принимает аудиозапись звонка, строит структурированный транскрипт с временными метками и спикерами, затем запускает четыре LLM-агента: классификацию темы, контроль качества, compliance-проверку и суммаризацию.

Проект собран под требования тестового задания: FastAPI API, OpenWebUI Pipelines, локальный ASR на `faster-whisper`, Docker Compose, тестовые аудиофайлы, `.env`-конфигурация, JSON-логирование и pytest-тесты.

## Архитектура

```text
OpenWebUI chat
  -> OpenWebUI Pipeline: pipeline.py
  -> FastAPI: POST /analyze
  -> faster-whisper ASR
  -> normalizer + basic diarizer
  -> AnalysisOrchestrator
  -> classifier / quality / compliance / summarizer
  -> JSON for API and Markdown for OpenWebUI
```

Основной backend находится в `app/`, ASR-компоненты в `asr/`, LLM-агенты в `agents/`, OpenWebUI Pipeline в корневом `pipeline.py`.

Для оркестрации выбран простой supervisor-паттерн в `app/orchestrator.py`: четыре агента запускаются параллельно через `ThreadPoolExecutor`, получают один и тот же транскрипт и возвращают независимые части итогового анализа. Для прототипа это проще и надёжнее, чем добавлять LangGraph: граф здесь линейный, без ветвления состояния, а требования закрываются прозрачной координацией и JSON-логированием входа/выхода каждого агента.

## Возможности

- `POST /analyze` принимает аудиофайл через `multipart/form-data`.
- `POST /analyze-url` принимает JSON с URL аудиофайла.
- `GET /health` возвращает статус API.
- OpenWebUI Pipeline ищет загруженный аудиофайл в хранилище OpenWebUI и отправляет его в API.
- ASR использует `faster-whisper` с моделью из `WHISPER_MODEL`.
- Поддерживаются WAV, MP3, OGG, M4A, AIFF, FLAC на уровне Pipeline/API входа.
- Базовая диаризация размечает реплики как `Оператор` / `Клиент`.
- Результат содержит транскрипт, сегменты, классификацию, quality score, compliance, summary и action items.

## Быстрый старт

Создайте `.env` из примера:

```bash
cp .env.example .env
```

Заполните `GEMINI_API_KEY`. По умолчанию используется OpenAI-совместимый Gemini endpoint:

```env
GEMINI_API_KEY=your-gemini-api-key
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.6-flash
WHISPER_MODEL=medium
```

Запуск всего стека:

```bash
docker compose up --build
```

После запуска доступны:

- FastAPI: `http://localhost:8000`
- OpenWebUI: `http://localhost:3000`
- OpenWebUI Pipelines: `http://localhost:9099`

Проверка API:

```bash
curl -s http://localhost:8000/health
```

Пример анализа файла:

```bash
curl -s -F "file=@test_data/calls/call_03_transfer.wav" http://localhost:8000/analyze
```

Пример анализа по URL:

```bash
curl -s \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/call.wav"}' \
  http://localhost:8000/analyze-url
```

## Формат ответа API

```json
{
  "transcript": "Оператор: ...\nКлиент: ...",
  "segments": [
    {
      "start": 0.0,
      "end": 4.2,
      "raw_text": "...",
      "text": "...",
      "speaker": "Оператор"
    }
  ],
  "analysis": {
    "classification": {
      "topic": "переводы",
      "priority": "medium"
    },
    "quality_score": {
      "total": 100,
      "checklist": {
        "greeting": true,
        "need_detection": true,
        "solution_provided": true,
        "farewell": true
      }
    },
    "compliance": {
      "passed": true,
      "issues": []
    },
    "summary": "...",
    "action_items": ["..."]
  }
}
```

## LLM-агенты

| Агент | Файл | Что делает |
|---|---|---|
| Классификатор | `agents/classifier.py` | Определяет тему: кредиты, карты, переводы, жалобы или не определено; выставляет priority. |
| Агент качества | `agents/quality.py` | Проверяет приветствие, выявление потребности, предложенное решение и завершение разговора. |
| Compliance | `agents/compliance.py` | Ищет запрещённые обещания, просьбы сообщить секретные данные, некорректные банковские формулировки. |
| Суммаризатор | `agents/summarizer.py` | Делает резюме на 3-5 предложений и список следующих действий. |

Каждый агент возвращает строго JSON. Ответы валидируются в коде, поэтому некорректный LLM-ответ не проходит дальше молча.

## ASR и диаризация

`asr/transcriber.py` использует `faster-whisper`. Сегменты нормализуются в `asr/normalizer.py`: чистятся базовые пробелы и варианты названия МТБанка. `asr/diarizer.py` реализует лёгкую доменную диаризацию по структуре звонка и речевым признакам оператора/клиента.

Это осознанный компромисс для прототипа: полноценная speaker diarization модель тяжелее и сложнее для локального Docker-демо, а в тестовом задании требуется базовая разметка `Оператор` / `Клиент`.

## Тестовые данные

В репозитории есть 5 синтетических русскоязычных звонков в `test_data/calls/` и эталонные тексты в `test_data/references/`.

| Файл | Тема | Sample rate | Длительность | Особенность |
|---|---|---:|---:|---|
| `call_01_credit.wav` | кредиты | 16 kHz | 73.8 сек | консультация по кредиту |
| `call_02_card.wav` | карты | 16 kHz | 94.6 сек | проблема с оплатой картой |
| `call_03_transfer.wav` | переводы | 16 kHz | 90.9 сек | межбанковский перевод |
| `call_04_complaint.wav` | жалобы | 16 kHz | 91.6 сек | жалоба на обслуживание |
| `call_05_security_8khz.wav` | безопасность | 8 kHz | 93.7 сек | телефонное качество |

Общая длительность: около 7 минут 24 секунд. Все файлы являются диалогами оператор/клиент и подходят для проверки ASR, диаризации и агентов.

Smoke-проверка всех аудио через API:

```bash
for f in test_data/calls/*.wav; do
  echo "$f"
  curl -s -F "file=@$f" http://localhost:8000/analyze \
    | python3 -m json.tool >/dev/null \
    && echo OK
done
```

Последний локальный прогон: все 5 файлов вернули корректный JSON.

WER по эталонным расшифровкам на локальном прогоне:

| Файл | WER |
|---|---:|
| `call_01_credit.wav` | `0.2342` |
| `call_02_card.wav` | `0.2133` |
| `call_03_transfer.wav` | `0.2786` |
| `call_04_complaint.wav` | `0.2057` |
| `call_05_security_8khz.wav` | `0.1818` |

Средний WER по набору: `0.2227`.

## Тесты

Локальный запуск:

```bash
pytest -q
```

Текущее состояние:

```text
34 passed
```

Покрыты:

- агенты classifier, quality, compliance, summarizer;
- pipeline сборки транскрипта и анализа;
- API endpoints;
- orchestrator;
- diarizer;
- normalizer.

## JSON-логирование

`app/json_logging.py` пишет структурированные JSON-события для каждого агента:

- `agent.input`;
- `agent.output`;
- `agent.error`.

В логах фиксируются имя агента, событие, payload и длительность выполнения.

## Docker Compose

`docker-compose.yml` поднимает три сервиса:

| Сервис | Назначение | Порт |
|---|---|---:|
| `api` | FastAPI + ASR + агенты | `8000` |
| `pipelines` | OpenWebUI Pipelines runtime | `9099` |
| `openwebui` | веб-интерфейс OpenWebUI | `3000` |

Whisper cache вынесен в Docker volume `whisper_cache`, чтобы модель не скачивалась заново при каждом запуске.

## Ограничения прототипа

- Диаризация базовая и доменная, без отдельной модели распознавания голосов.
- Первый запуск может быть медленным из-за загрузки Whisper-модели.
- Качество LLM-аналитики зависит от доступности и квот выбранной OpenAI-совместимой модели.
- Для публичной сдачи нужен HTTPS URL демо и публичный GitHub-репозиторий.
