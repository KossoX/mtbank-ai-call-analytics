import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, HttpUrl
from starlette.responses import Response

from agents.trends import TrendsAgent
from app.config import get_settings
from app.json_logging import log_agent_event
from app.llm import LLMQuotaExceededError
from app.metrics import (
    REALTIME_CHUNK_PROCESSING_SECONDS,
    REALTIME_CHUNKS_TOTAL,
    TREND_BATCHES_TOTAL,
)
from app.pipeline import AudioAnalysisPipeline
from asr.streaming import StreamingTranscriber


MAX_AUDIO_BYTES = 50 * 1024 * 1024
MAX_BATCH_FILES = 10
MIN_BATCH_FILES = 2


class AnalysisPipelineProtocol(Protocol):
    def analyze(self, audio_path: str | Path) -> dict[str, Any]:
        ...


class TrendsAgentProtocol(Protocol):
    def analyze(
        self,
        calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ...


class StreamingTranscriberProtocol(Protocol):
    def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate: int,
        language: str = "ru",
    ) -> list[dict[str, Any]]:
        ...


class AnalyzeUrlRequest(BaseModel):
    url: HttpUrl


def create_app(
    pipeline: AnalysisPipelineProtocol | None = None,
    trends_agent: TrendsAgentProtocol | None = None,
    streaming_transcriber: StreamingTranscriberProtocol | None = None,
) -> FastAPI:
    active_pipeline = pipeline
    active_trends_agent = trends_agent
    active_streaming_transcriber = streaming_transcriber

    def get_pipeline() -> AnalysisPipelineProtocol:
        nonlocal active_pipeline

        if active_pipeline is None:
            active_pipeline = AudioAnalysisPipeline()

        return active_pipeline

    def get_trends_agent() -> TrendsAgentProtocol:
        nonlocal active_trends_agent

        if active_trends_agent is None:
            active_trends_agent = TrendsAgent()

        return active_trends_agent

    def get_streaming_transcriber() -> StreamingTranscriberProtocol:
        nonlocal active_streaming_transcriber

        if active_streaming_transcriber is None:
            active_streaming_transcriber = StreamingTranscriber()

        return active_streaming_transcriber

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        transcriber = get_streaming_transcriber()
        preload = getattr(transcriber, "preload", None)

        if callable(preload):
            await run_in_threadpool(preload)

        yield

    app = FastAPI(
        title="MTB Bank Call Analytics",
        version="1.2.0",
        description=(
            "ASR, multi-agent call analytics, real-time transcription "
            "and multi-call trend analysis."
        ),
        lifespan=lifespan,
    )
    allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ALLOW_ORIGIN",
            "http://localhost:3000",
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def analyze_path(audio_path: Path) -> dict[str, Any]:
        try:
            return get_pipeline().analyze(audio_path)
        except LLMQuotaExceededError as error:
            raise HTTPException(
                status_code=429,
                detail=(
                    "LLM request quota exceeded. "
                    "Please retry later or configure another model."
                ),
            ) from error
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.post("/analyze")
    async def analyze_audio(
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        audio_bytes = await _read_upload(file)
        temporary_path = _write_temporary_audio(
            audio_bytes,
            file.filename,
        )

        try:
            return await run_in_threadpool(
                analyze_path,
                temporary_path,
            )
        finally:
            temporary_path.unlink(missing_ok=True)

    @app.post("/analyze-batch")
    async def analyze_audio_batch(
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        if not MIN_BATCH_FILES <= len(files) <= MAX_BATCH_FILES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Upload between {MIN_BATCH_FILES} and "
                    f"{MAX_BATCH_FILES} audio files."
                ),
            )

        calls: list[dict[str, Any]] = []

        for index, file in enumerate(files):
            audio_bytes = await _read_upload(file)
            temporary_path = _write_temporary_audio(
                audio_bytes,
                file.filename,
            )

            try:
                result = await run_in_threadpool(
                    analyze_path,
                    temporary_path,
                )
            finally:
                temporary_path.unlink(missing_ok=True)

            calls.append(
                {
                    "index": index,
                    "filename": file.filename,
                    **result,
                }
            )

        trend_started_at = perf_counter()
        log_agent_event(
            agent="trends",
            event="agent.input",
            payload={"calls": calls},
        )

        try:
            trends = await run_in_threadpool(
                get_trends_agent().analyze,
                calls,
            )
        except LLMQuotaExceededError as error:
            log_agent_event(
                agent="trends",
                event="agent.error",
                payload={
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                duration_ms=(
                    perf_counter() - trend_started_at
                ) * 1000,
            )
            raise HTTPException(
                status_code=429,
                detail=(
                    "LLM request quota exceeded during trend analysis."
                ),
            ) from error
        except ValueError as error:
            log_agent_event(
                agent="trends",
                event="agent.error",
                payload={
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                duration_ms=(
                    perf_counter() - trend_started_at
                ) * 1000,
            )
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

        log_agent_event(
            agent="trends",
            event="agent.output",
            payload=trends,
            duration_ms=(
                perf_counter() - trend_started_at
            ) * 1000,
        )
        TREND_BATCHES_TOTAL.inc()

        return {
            "calls": calls,
            "trends": trends,
        }

    @app.post("/analyze-url")
    async def analyze_audio_url(
        payload: AnalyzeUrlRequest,
    ) -> dict[str, Any]:
        parsed_url = urlparse(str(payload.url))

        if parsed_url.scheme not in {"http", "https"}:
            raise HTTPException(
                status_code=400,
                detail="Only HTTP and HTTPS URLs are supported.",
            )

        try:
            audio_bytes = await run_in_threadpool(
                _download_audio,
                str(payload.url),
            )
        except HTTPError as error:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not download audio URL: HTTP {error.code}."
                ),
            ) from error
        except (URLError, TimeoutError) as error:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not download audio from the provided URL."
                ),
            ) from error

        suffix_name = Path(parsed_url.path).name or "download.wav"
        temporary_path = _write_temporary_audio(
            audio_bytes,
            suffix_name,
        )

        try:
            return await run_in_threadpool(
                analyze_path,
                temporary_path,
            )
        finally:
            temporary_path.unlink(missing_ok=True)

    @app.websocket("/ws/transcribe")
    async def transcribe_realtime(websocket: WebSocket) -> None:
        await websocket.accept()

        settings = get_settings()
        sample_rate = 16000
        language = "ru"
        chunk_seconds = settings.realtime_chunk_seconds
        bytes_per_sample = 2
        buffer = bytearray()
        audio_offset = 0.0
        chunk_index = 0

        await websocket.send_json(
            {
                "type": "ready",
                "format": "pcm_s16le",
                "channels": 1,
                "sample_rate": sample_rate,
                "chunk_seconds": chunk_seconds,
                "latency_target_ms": 3000,
            }
        )

        async def process_chunk(
            pcm_bytes: bytes,
            *,
            final: bool,
        ) -> None:
            nonlocal audio_offset, chunk_index

            if not pcm_bytes:
                return

            audio_seconds = (
                len(pcm_bytes)
                / (sample_rate * bytes_per_sample)
            )
            started_at = perf_counter()

            try:
                segments = await run_in_threadpool(
                    get_streaming_transcriber().transcribe_pcm,
                    pcm_bytes,
                    sample_rate=sample_rate,
                    language=language,
                )
            except Exception as error:
                REALTIME_CHUNKS_TOTAL.labels(status="error").inc()
                await websocket.send_json(
                    {
                        "type": "error",
                        "detail": str(error),
                    }
                )
                return

            processing_seconds = perf_counter() - started_at
            REALTIME_CHUNKS_TOTAL.labels(status="success").inc()
            REALTIME_CHUNK_PROCESSING_SECONDS.observe(
                processing_seconds
            )

            adjusted_segments: list[dict[str, Any]] = []

            for segment in segments:
                adjusted = dict(segment)
                adjusted["start"] = round(
                    float(adjusted.get("start", 0))
                    + audio_offset,
                    3,
                )
                adjusted["end"] = round(
                    float(adjusted.get("end", 0))
                    + audio_offset,
                    3,
                )
                adjusted_segments.append(adjusted)

            await websocket.send_json(
                {
                    "type": "final" if final else "partial",
                    "chunk_index": chunk_index,
                    "audio_offset": round(audio_offset, 3),
                    "audio_seconds": round(audio_seconds, 3),
                    "processing_ms": round(
                        processing_seconds * 1000,
                        2,
                    ),
                    "latency_target_met": (
                        processing_seconds < 3.0
                    ),
                    "segments": adjusted_segments,
                    "text": " ".join(
                        str(segment.get("text", ""))
                        for segment in adjusted_segments
                    ).strip(),
                }
            )
            audio_offset += audio_seconds
            chunk_index += 1

        try:
            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                text_message = message.get("text")
                binary_message = message.get("bytes")

                if text_message is not None:
                    try:
                        command = json.loads(text_message)
                    except json.JSONDecodeError:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "detail": "Control message must be JSON.",
                            }
                        )
                        continue

                    command_type = command.get("type")

                    if command_type == "start":
                        requested_sample_rate = command.get(
                            "sample_rate",
                            sample_rate,
                        )
                        if requested_sample_rate not in {
                            8000,
                            16000,
                            24000,
                            48000,
                        }:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "detail": (
                                        "Unsupported sample_rate."
                                    ),
                                }
                            )
                            continue

                        sample_rate = int(requested_sample_rate)
                        language = str(
                            command.get("language", "ru")
                        )
                        await websocket.send_json(
                            {
                                "type": "started",
                                "sample_rate": sample_rate,
                                "language": language,
                            }
                        )
                        continue

                    if command_type == "flush":
                        pending = bytes(buffer)
                        buffer.clear()
                        await process_chunk(
                            pending,
                            final=False,
                        )
                        continue

                    if command_type == "stop":
                        pending = bytes(buffer)
                        buffer.clear()
                        await process_chunk(
                            pending,
                            final=True,
                        )
                        await websocket.send_json(
                            {
                                "type": "completed",
                                "chunks": chunk_index,
                                "audio_seconds": round(
                                    audio_offset,
                                    3,
                                ),
                            }
                        )
                        await websocket.close(code=1000)
                        break

                    await websocket.send_json(
                        {
                            "type": "error",
                            "detail": "Unknown control message type.",
                        }
                    )
                    continue

                if binary_message is not None:
                    buffer.extend(binary_message)
                    chunk_size = int(
                        sample_rate
                        * bytes_per_sample
                        * chunk_seconds
                    )

                    while len(buffer) >= chunk_size:
                        current = bytes(buffer[:chunk_size])
                        del buffer[:chunk_size]
                        await process_chunk(
                            current,
                            final=False,
                        )
        except WebSocketDisconnect:
            return

    return app


async def _read_upload(file: UploadFile) -> bytes:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio file must have a filename.",
        )

    audio_bytes = await file.read(MAX_AUDIO_BYTES + 1)

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio file is empty.",
        )

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Uploaded audio file is too large.",
        )

    return audio_bytes


def _write_temporary_audio(
    audio_bytes: bytes,
    filename: str | None,
) -> Path:
    suffix = Path(filename or "audio.wav").suffix or ".wav"

    with NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as temporary_file:
        temporary_file.write(audio_bytes)
        return Path(temporary_file.name)


def _download_audio(url: str) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "MTB-Bank-Call-Analytics/1.2"},
    )

    with urlopen(request, timeout=30) as response:
        audio_bytes = response.read(MAX_AUDIO_BYTES + 1)

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Audio file downloaded from URL is empty.",
        )

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Audio file downloaded from URL is too large.",
        )

    return audio_bytes


app = create_app()
