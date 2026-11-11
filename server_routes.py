"""HTTP routes for Snart nodes (the staged prompt builder's Suggest button)."""

import asyncio
from typing import Any, Dict, List

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - only importable inside ComfyUI
    web = None
    PromptServer = None

from .prompt_enhancer import suggest_options
from .llm_backend import LLMBackend


def _register(server) -> None:
    @server.routes.post("/snart/staged/suggest")
    async def handler(request):
        try:
            payload: Dict[str, Any] = await request.json()
        except Exception:
            payload = {}

        prompt = str(payload.get("prompt") or "")
        chosen: List[str] = [str(c) for c in (payload.get("chosen") or []) if str(c).strip()]
        step = int(payload.get("step") or 1)
        num = int(payload.get("num_suggestions") or 2)
        broadness = str(payload.get("broadness") or "simple")
        backend = str(payload.get("llm_backend") or "kindi")
        endpoint = str(payload.get("api_endpoint") or "")
        model_name = str(payload.get("model_name") or "")
        temperature = float(payload.get("temperature") or 0.7)

        def work():
            try:
                llm = LLMBackend(
                    backend_type=backend,
                    endpoint=endpoint,
                    model_name=model_name.strip() or None,
                    temperature=temperature,
                    api_key=None,
                )
                options = suggest_options(llm, prompt, chosen, step, num, broadness)
                if not options:
                    return {"success": False, "options": [], "error": "The LLM returned no options."}
                return {"success": True, "options": options, "error": None}
            except Exception as exc:  # never let the route 500
                return {"success": False, "options": [], "error": str(exc)}

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, work)
        return web.json_response(result)

    print("[Snart] Registered POST /snart/staged/suggest for the staged prompt builder")


if PromptServer is not None:
    try:
        _register(PromptServer.instance)
    except Exception as exc:  # pragma: no cover
        print(f"[Snart] Could not register staged-suggest route: {exc}")
