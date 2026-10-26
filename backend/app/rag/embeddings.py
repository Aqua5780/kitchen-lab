"""Text -> vector, via the configured embedding provider."""

from openai import OpenAI

from app.core.config import settings

_client: OpenAI | object | None = None


def get_client() -> OpenAI | object:
    global _client
    if _client is None:
        if settings.llm_provider == "gemini":
            from google import genai

            if not settings.gemini_api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Add it to the .env file at "
                    "the project root (see .env.example)."
                )
            _client = genai.Client(api_key=settings.gemini_api_key)
        elif settings.llm_provider == "openai":
            if not settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it to the .env file at "
                    "the project root (see .env.example)."
                )
            _client = OpenAI(api_key=settings.openai_api_key)
        else:
            raise RuntimeError(
                f"Unsupported LLM_PROVIDER={settings.llm_provider!r}. "
                "Use 'gemini' or 'openai'."
            )
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in one API call (batching is cheaper/faster)."""
    client = get_client()
    if settings.llm_provider == "gemini":
        from google.genai import types

        response = client.models.embed_content(
            model=settings.embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=1536),
        )
        return [embedding.values for embedding in response.embeddings]

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
