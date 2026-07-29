import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    llm_base_url: str
    llm_model: str
    whisper_model: str
    realtime_whisper_model: str
    realtime_chunk_seconds: float


def get_settings() -> Settings:
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Create a local .env file in the project root."
        )

    realtime_chunk_seconds = float(os.getenv("REALTIME_CHUNK_SECONDS", "1.0"))

    if not 0.5 <= realtime_chunk_seconds <= 10.0:
        raise RuntimeError("REALTIME_CHUNK_SECONDS must be between 0.5 and 10.")

    return Settings(
        gemini_api_key=gemini_api_key,
        llm_base_url=os.getenv(
            "LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        llm_model=os.getenv(
            "LLM_MODEL",
            "gemini-3.5-flash-lite",
        ),
        whisper_model=os.getenv("WHISPER_MODEL", "medium"),
        realtime_whisper_model=os.getenv(
            "REALTIME_WHISPER_MODEL",
            "tiny",
        ),
        realtime_chunk_seconds=realtime_chunk_seconds,
    )
