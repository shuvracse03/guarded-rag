import json

import httpx

from app.errors import ApiError


class OllamaLLM:
    """Local open-weight model served by Ollama (default: llama3.2 3B; qwen2.5:7b-instruct also tested)."""

    def __init__(self, base_url: str, model: str, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def ping(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/api/tags", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": 7, "num_ctx": 4096},
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except httpx.TimeoutException as exc:
            raise ApiError(504, "llm_timeout", "The language model timed out") from exc
        except httpx.ConnectError as exc:
            raise ApiError(503, "llm_unavailable", "Cannot reach the local Ollama server") from exc
        except httpx.HTTPStatusError as exc:
            raise ApiError(502, "llm_error", f"Ollama returned HTTP {exc.response.status_code}") from exc
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise ApiError(502, "llm_bad_response", "Unexpected response from Ollama") from exc
