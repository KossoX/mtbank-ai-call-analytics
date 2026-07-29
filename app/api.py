from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl

from app.llm import LLMQuotaExceededError
from app.pipeline import AudioAnalysisPipeline


MAX_AUDIO_BYTES = 50 * 1024 * 1024


class AnalyzeUrlRequest(BaseModel):
    url: HttpUrl


def create_app(
    pipeline: AudioAnalysisPipeline | None = None,
) -> FastAPI:
    app = FastAPI(
        title="MTB Bank Call Analytics",
        version="0.1.0",
    )

    def analyze_path(audio_path: Path) -> dict[str, Any]:
        try:
            active_pipeline = pipeline or AudioAnalysisPipeline()
            return active_pipeline.analyze(audio_path)

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

    @app.post("/analyze")
    async def analyze_audio(
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Audio file must have a filename.",
            )

        audio_bytes = await file.read()

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

        suffix = Path(file.filename).suffix or ".wav"
        temporary_path: Path | None = None

        try:
            with NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temporary_file:
                temporary_file.write(audio_bytes)
                temporary_path = Path(temporary_file.name)

            return analyze_path(temporary_path)

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

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

        suffix = Path(parsed_url.path).suffix or ".wav"
        temporary_path: Path | None = None

        try:
            request = Request(
                str(payload.url),
                headers={"User-Agent": "MTB-Bank-Call-Analytics/1.0"},
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

            with NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temporary_file:
                temporary_file.write(audio_bytes)
                temporary_path = Path(temporary_file.name)

            return analyze_path(temporary_path)

        except HTTPException:
            raise

        except HTTPError as error:
            raise HTTPException(
                status_code=400,
                detail=f"Could not download audio URL: HTTP {error.code}.",
            ) from error

        except (URLError, TimeoutError) as error:
            raise HTTPException(
                status_code=400,
                detail="Could not download audio from the provided URL.",
            ) from error

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return app


app = create_app()