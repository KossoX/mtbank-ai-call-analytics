from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.llm import LLMQuotaExceededError
from app.pipeline import AudioAnalysisPipeline


def create_app(
    pipeline: AudioAnalysisPipeline | None = None,
) -> FastAPI:
    app = FastAPI(
        title="MTB Bank Call Analytics",
        version="0.1.0",
    )

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

        suffix = Path(file.filename).suffix or ".wav"
        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio file is empty.",
            )

        temporary_path: Path | None = None

        try:
            with NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temporary_file:
                temporary_file.write(audio_bytes)
                temporary_path = Path(temporary_file.name)

            active_pipeline = pipeline or AudioAnalysisPipeline()
            return active_pipeline.analyze(temporary_path)

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

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return app


app = create_app()