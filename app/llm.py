from collections.abc import Iterable
from typing import Protocol

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.config import get_settings


Message = ChatCompletionMessageParam


class LLMProvider(Protocol):
    def complete(
        self,
        messages: Iterable[ChatCompletionMessageParam],
    ) -> str:
        ...


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()

        self._client = OpenAI(
            api_key=settings.gemini_api_key,
            base_url=settings.llm_base_url,
            timeout=45.0,
        )
        self._model = settings.llm_model

    def complete(self, messages: Iterable[Message]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=list(messages),
            temperature=0,
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("LLM returned an empty response.")

        return content.strip()