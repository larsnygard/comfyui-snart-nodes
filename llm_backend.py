"""OpenAI-compatible chat client for Snart ComfyUI nodes.

Original implementation. Speaks /chat/completions and /models to local servers
(Strata, Kindi, or any OpenAI-compatible endpoint) and to hosted providers
(OpenAI, DeepSeek, Mistral, Gemini, OpenRouter, Groq, Together).

API keys are read from this pack's config.json (gitignored) or an environment
variable - never from a node widget - so they cannot leak into saved workflow
JSON or generated-image metadata.
"""

import json
import os
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

# key -> spec. "env" is the environment variable that may hold the key;
# "key_optional" marks local servers that accept requests without a key.
BACKENDS: Dict[str, Dict[str, Any]] = {
    "openai":     {"label": "OpenAI",        "base_url": "https://api.openai.com/v1",                       "env": "OPENAI_API_KEY",   "model": "gpt-4o-mini"},
    "deepseek":   {"label": "DeepSeek",      "base_url": "https://api.deepseek.com/v1",                     "env": "DEEPSEEK_API_KEY", "model": "deepseek-chat"},
    "mistral":    {"label": "Mistral",       "base_url": "https://api.mistral.ai/v1",                       "env": "MISTRAL_API_KEY",  "model": "mistral-large-latest"},
    "gemini":     {"label": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "env": "GEMINI_API_KEY", "model": "gemini-2.0-flash"},
    "openrouter": {"label": "OpenRouter",    "base_url": "https://openrouter.ai/api/v1",                    "env": "OPENROUTER_API_KEY", "model": "openai/gpt-4o-mini"},
    "groq":       {"label": "Groq",          "base_url": "https://api.groq.com/openai/v1",                  "env": "GROQ_API_KEY",     "model": "llama-3.3-70b-versatile"},
    "together":   {"label": "Together AI",   "base_url": "https://api.together.xyz/v1",                     "env": "TOGETHER_API_KEY", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    "strata":     {"label": "Strata (local)", "base_url": "http://127.0.0.1:8080/v1", "env": "STRATA_API_KEY", "model": "swift-1.5-iq3_s", "key_optional": True},
    "kindi":      {"label": "Kindi (local)",  "base_url": "http://127.0.0.1:8093/v1", "env": "KINDI_API_KEY",  "model": "bonsai2-27b",    "key_optional": True},
    "custom":     {"label": "Custom (OpenAI-compatible)", "base_url": "http://localhost:1234/v1", "env": "CUSTOM_API_KEY", "model": "local-model", "key_optional": True},
}

# Order shown in the node dropdowns: local servers first, then hosted.
OPTION_ORDER = (
    "strata", "kindi", "custom",
    "openai", "deepseek", "mistral", "gemini", "openrouter", "groq", "together",
)

# Endpoint values that still mean "the node default"; replaced by a backend's
# base_url so selecting a backend works without editing the endpoint field.
_DEFAULT_ENDPOINTS = {
    "http://localhost:1234/v1",
    "http://localhost:1234",
    "http://127.0.0.1:1234/v1",
    "http://127.0.0.1:8080/v1",
    "http://127.0.0.1:8093/v1",
}

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


# --- config ---------------------------------------------------------------


def _load_config() -> Dict[str, Any]:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _config_key(backend: str) -> Optional[str]:
    entry = (_load_config().get("backends") or {}).get(backend)
    if isinstance(entry, dict):
        return entry.get("api_key")
    return None


def _is_placeholder(value: Optional[str]) -> bool:
    """True for values that are clearly not a real API key."""
    text = (value or "").strip().lower()
    if not text:
        return True
    if text.startswith("your_") or text.startswith("your-"):
        return True
    return text in {"none", "changeme", "not-needed", "dummy"}


# --- public helpers -------------------------------------------------------


def get_backend_options() -> List[str]:
    return list(OPTION_ORDER)


def get_backend_tooltip() -> str:
    lines = []
    for key in OPTION_ORDER:
        spec = BACKENDS[key]
        if spec.get("key_optional"):
            lines.append(f"{key}: {spec['label']} (local; API key optional)")
        else:
            lines.append(f"{key}: {spec['label']} (cloud; requires an API key)")
    return "\n".join(lines)


def resolve_endpoint(backend: str, endpoint: Optional[str] = None) -> str:
    spec = BACKENDS.get((backend or "").lower())
    value = (endpoint or "").strip()
    if spec and (not value or value.rstrip("/") in _DEFAULT_ENDPOINTS):
        value = spec["base_url"]
    return value.rstrip("/")


def resolve_api_key(backend: str, explicit: Optional[str] = None) -> Optional[str]:
    spec = BACKENDS.get((backend or "").lower())
    if not spec:
        return None
    for candidate in (explicit, _config_key(backend), os.environ.get(spec["env"])):
        if isinstance(candidate, str) and not _is_placeholder(candidate):
            return candidate.strip()
    return None


def list_models(backend: str, endpoint: Optional[str] = None,
                api_key: Optional[str] = None, timeout: float = 15.0) -> Dict[str, Any]:
    """Query a backend's /models endpoint. Returns {success, models, error}."""
    spec = BACKENDS.get((backend or "").lower())
    if not spec:
        return {"success": False, "models": [], "error": f"Unknown backend: {backend}"}

    url = resolve_endpoint(backend, endpoint)
    key = resolve_api_key(backend, api_key)
    if not key and not spec.get("key_optional"):
        return {"success": False, "models": [], "error": f"{spec['label']} needs an API key."}

    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        response = requests.get(f"{url}/models", headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        return {"success": False, "models": [], "error": f"Timed out reaching {url}."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "models": [], "error": f"Cannot reach {url}."}
    except requests.exceptions.RequestException as exc:
        return {"success": False, "models": [], "error": str(exc)}

    if response.status_code in (401, 403):
        return {"success": False, "models": [], "error": f"{spec['label']} rejected the request (HTTP {response.status_code}). Check the API key."}
    if response.status_code >= 400:
        return {"success": False, "models": [], "error": f"{spec['label']} error (HTTP {response.status_code})."}

    try:
        data = response.json()
    except ValueError:
        return {"success": False, "models": [], "error": "Backend returned non-JSON."}

    names = set()
    for entry in (data.get("data") or data.get("models") or []):
        if isinstance(entry, str):
            names.add(entry)
        elif isinstance(entry, dict):
            ident = entry.get("id") or entry.get("model") or entry.get("name")
            if ident:
                names.add(str(ident))
    return {"success": True, "models": sorted(names), "error": None}


# --- client ---------------------------------------------------------------


class LLMBackend:
    """A configured chat client for one backend and model."""

    def __init__(self, backend_type: str, endpoint: str = "", model_name: Optional[str] = None,
                 temperature: float = 0.7, api_key: Optional[str] = None):
        self.backend = (backend_type or "").lower()
        self.endpoint = resolve_endpoint(self.backend, endpoint)
        self.temperature = float(temperature)
        self.api_key = resolve_api_key(self.backend, api_key)
        self.model = (model_name or "").strip() or self._detect_model()

    def _detect_model(self) -> str:
        result = list_models(self.backend, self.endpoint, self.api_key)
        models = result.get("models") or []
        if result.get("success") and models:
            return models[0]
        spec = BACKENDS.get(self.backend, {})
        return spec.get("model", "default")

    def send_prompt(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Send a chat prompt. Returns {success, response, error}."""
        spec = BACKENDS.get(self.backend, {})
        label = spec.get("label", self.backend)

        if not self.api_key and not spec.get("key_optional"):
            return {
                "success": False,
                "response": "",
                "error": f"{label} needs an API key (config.json or the {spec.get('env')} environment variable).",
            }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(f"{self.endpoint}/chat/completions", json=payload, headers=headers, timeout=300)
        except requests.exceptions.Timeout:
            return {"success": False, "response": "", "error": f"{label} timed out."}
        except requests.exceptions.ConnectionError:
            return {"success": False, "response": "", "error": f"Cannot reach {label} at {self.endpoint}."}
        except requests.exceptions.RequestException as exc:
            return {"success": False, "response": "", "error": f"{label} request failed: {exc}"}

        if response.status_code in (401, 403):
            return {"success": False, "response": "", "error": f"{label} rejected the request (HTTP {response.status_code}). Check the API key."}
        if response.status_code == 402:
            return {"success": False, "response": "", "error": f"{label} reports an empty balance (HTTP 402)."}
        if response.status_code == 429:
            return {"success": False, "response": "", "error": f"{label} rate limit reached (HTTP 429). Retry later."}
        if response.status_code >= 400:
            return {"success": False, "response": "", "error": f"{label} error (HTTP {response.status_code})."}

        try:
            data = response.json()
        except ValueError:
            return {"success": False, "response": "", "error": f"{label} returned non-JSON."}

        choices = data.get("choices")
        if not choices:
            err = data.get("error") or {}
            message = err.get("message") if isinstance(err, dict) else str(err or "no 'choices' in response")
            return {"success": False, "response": "", "error": f"{label}: {message}"}

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            # reasoning models return their chain of thought separately
            content = message.get("reasoning_content")
        return {"success": True, "response": (content or "").strip(), "error": None}
