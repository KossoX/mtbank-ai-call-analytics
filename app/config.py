from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    llm_base_url: str
    llm_model: str
    whisper_model: str


def get_settings() -> Settings:
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Create a local .env file in the project root."
        )

    return Settings(
        gemini_api_key=gemini_api_key,
        llm_base_url=os.getenv(
            "LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        llm_model=os.getenv("LLM_MODEL", "gemini-3.6-flash"),
        whisper_model=os.getenv("WHISPER_MODEL", "medium"),
    )