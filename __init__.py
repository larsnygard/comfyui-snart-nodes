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
from .prompt_enhancer import SnartStyleSuggester, SnartPromptEnhancer

NODE_CLASS_MAPPINGS = {
    "LoraStacker": LoraStacker,
    "SnartEmptyLatent": SnartEmptyLatent,
    "SnartStyleSuggester": SnartStyleSuggester,
    "SnartPromptEnhancer": SnartPromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraStacker": "LoRA Stacker",
    "SnartEmptyLatent": "Empty Latent",
    "SnartStyleSuggester": "Style Suggester",
    "SnartPromptEnhancer": "Prompt Enhancer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
