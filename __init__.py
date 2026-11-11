"""
Snart nodes for ComfyUI.

Lars' own node pack. Currently:

  - SnartLoraStacker: stack multiple LoRAs in one node (manual / random /
    LLM-selected, with optional checkpoint base-model matching).

The LLM selection mode uses the pack's llm_backend, which talks to local
OpenAI-compatible servers (Strata, Kindi) or hosted providers. API keys are
read from this pack's config.json (gitignored) or environment variables.
"""

from .lora_stacker import LoraStacker
from .empty_latent import SnartEmptyLatent
from .prompt_enhancer import SnartStagedPrompter, SnartPromptBuilder, SnartPromptEnhancer

NODE_CLASS_MAPPINGS = {
    "LoraStacker": LoraStacker,
    "SnartEmptyLatent": SnartEmptyLatent,
    "SnartStagedPrompter": SnartStagedPrompter,
    "SnartPromptBuilder": SnartPromptBuilder,
    "SnartPromptEnhancer": SnartPromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraStacker": "LoRA Stacker",
    "SnartEmptyLatent": "Empty Latent",
    "SnartStagedPrompter": "Staged Prompt Builder",
    "SnartPromptBuilder": "Prompt Builder",
    "SnartPromptEnhancer": "Prompt Enhancer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Frontend assets: web/js/staged_suggest.js adds the "Suggest" button + dropdown.
WEB_DIRECTORY = "./web"

# Registers POST /snart/staged/suggest (used by that button).
try:
    from . import server_routes  # noqa: F401
except Exception as _route_error:  # pragma: no cover
    print(f"[Snart] Staged-suggest route unavailable: {_route_error}")
