"""OpenAI Responses-shaped adapter for a loopback Ollama text server."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from .project_env import project_secret


class LocalTextError(ValueError):
    """The requested local text backend is not ready."""


def local_text_model(explicit: str | None = None) -> str:
    model = (project_secret("LOCAL_TEXT_MODEL", explicit) or "").strip()
    if not model:
        raise LocalTextError("LOCAL_TEXT_MODEL is required for TEXT_PROVIDER=local")
    return model


def local_text_base_url(explicit: str | None = None) -> str:
    url = (project_secret("LOCAL_TEXT_BASE_URL", explicit) or "http://127.0.0.1:11434/v1").strip()
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise LocalTextError("LOCAL_TEXT_BASE_URL must be a loopback HTTP address")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise LocalTextError("LOCAL_TEXT_BASE_URL must not contain credentials or query parameters")
    return url.rstrip("/") + "/"


class _ResponsesAdapter:
    def __init__(self, openai_client: Any) -> None:
        self._client = openai_client

    def create(self, **request: Any) -> Any:
        format_spec = request["text"]["format"]
        if format_spec.get("type") != "json_schema":
            raise LocalTextError("local text requests require a JSON schema")
        response = self._client.chat.completions.create(
            model=request["model"],
            messages=request["input"],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": format_spec["name"],
                    "schema": format_spec["schema"],
                    "strict": format_spec.get("strict", True),
                },
            },
            max_tokens=request["max_output_tokens"],
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return SimpleNamespace(
            output_text=content,
            usage={
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            },
        )


class LocalTextClient:
    """Expose the subset of Responses API used by the text stages."""

    def __init__(self, openai_client: Any) -> None:
        self.responses = _ResponsesAdapter(openai_client)


def load_local_text_client(base_url: str | None = None) -> LocalTextClient:
    url = local_text_base_url(base_url)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LocalTextError("local text mode needs the OpenAI-compatible SDK; install .[openai]") from exc
    return LocalTextClient(OpenAI(api_key="ollama", base_url=url, timeout=600.0))
