"""Narrow adapter around the local Ollama API."""

from __future__ import annotations

from collections.abc import Sequence

import ollama


class OllamaBackend:
    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = ollama.embed(model=model, input=list(texts))
        except Exception as error:  # Ollama exposes several transport/model errors.
            raise RuntimeError(_friendly_error(error, model)) from error
        return [list(vector) for vector in response.embeddings]

    def generate(
        self, messages: Sequence[dict[str, str]], model: str, temperature: float = 0.1
    ) -> str:
        try:
            response = ollama.chat(
                model=model,
                messages=list(messages),
                options={"temperature": temperature},
            )
        except Exception as error:
            raise RuntimeError(_friendly_error(error, model)) from error
        return response.message.content.strip()


def _friendly_error(error: Exception, model: str) -> str:
    return (
        f"Could not use Ollama model '{model}': {error}. "
        "Make sure Ollama is running and the model has been pulled."
    )
