"""Thin, provider-agnostic chat client.

Everything else in the app calls complete_json(); only this file knows the
provider is OpenAI. Swapping providers (Anthropic, local model) later means
changing this one function.
"""

import json

from app.core.config import settings
from app.rag.embeddings import get_client


def complete_json(system: str, user: str) -> dict:
    """Send a system+user prompt; get back parsed JSON.

    temperature=0.2: nearly deterministic. The explanation layer should be
    consistent, not creative.
    response_format json_object: the API guarantees syntactically valid JSON.
    """
    if settings.llm_provider == "gemini":
        from google.genai import types

        response = get_client().models.generate_content(
            model=settings.chat_model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        content = response.text
    elif settings.llm_provider == "openai":
        response = get_client().chat.completions.create(
            model=settings.chat_model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
    else:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER={settings.llm_provider!r}. "
            "Use 'gemini' or 'openai'."
        )

    if not content:
        raise RuntimeError("The LLM returned an empty response.")
    return json.loads(content)
