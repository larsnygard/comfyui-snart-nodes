"""Snart staged prompt enhancer.

Two nodes that build an image prompt in steps, with the LLM doing the creative
work instead of hardcoded style presets:

  - SnartStyleSuggester (stage 1): give it a prompt, it asks the LLM for a list
    of distinct style / art directions to choose from.
  - SnartPromptEnhancer  (stage 2): give it the prompt plus the style you picked
    (from stage 1 or your own words), it writes the final detailed prompt.

Works with any backend in llm_backend. API keys are read from config.json /
environment variables only - there is no api_key widget, so keys cannot leak.
"""

import re
from typing import Tuple

from .llm_backend import LLMBackend, get_backend_options, get_backend_tooltip


def _llm_widgets(default_backend: str = "kindi"):
    return {
        "llm_backend": (get_backend_options(), {"default": default_backend, "tooltip": get_backend_tooltip()}),
        "api_endpoint": ("STRING", {"default": "http://localhost:1234/v1"}),
        "model_name": ("STRING", {"default": "", "placeholder": "blank = auto-detect"}),
        "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
    }


def _client(llm_backend, api_endpoint, model_name, temperature):
    return LLMBackend(
        backend_type=llm_backend,
        endpoint=api_endpoint,
        model_name=(model_name or "").strip() or None,
        temperature=temperature,
        api_key=None,
    )


def _clean(text) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


class SnartStyleSuggester:
    """Stage 1: ask an LLM for style / art-direction suggestions for a prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Describe the image you want to create"}),
                "num_suggestions": ("INT", {"default": 5, "min": 1, "max": 12, "step": 1}),
                "focus": ("STRING", {"default": "", "placeholder": "Optional bias, e.g. 'photorealistic' or 'fantasy book cover'"}),
                **_llm_widgets(),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("suggestions", "prompt")
    FUNCTION = "suggest"
    CATEGORY = "Snart"

    def suggest(self, prompt, num_suggestions, focus, llm_backend, api_endpoint,
                model_name, temperature) -> Tuple[str, str]:
        prompt = (prompt or "").strip()
        if not prompt:
            return ("(enter a prompt first)", "")

        focus = (focus or "").strip()
        hint = f"\nBias the suggestions toward: {focus}." if focus else ""

        system = (
            "You are an art director helping choose a visual style for an image. "
            "Give concrete, distinct style directions, not vague advice."
        )
        user = (
            f"Image prompt:\n\"{prompt}\"\n\n"
            f"Suggest {num_suggestions} distinct visual styles or art directions for this image. "
            f"For each, give a short name and a one-line description of the look "
            f"(medium, lighting, palette, mood). Number them 1..{num_suggestions}.{hint}"
        )

        llm = _client(llm_backend, api_endpoint, model_name, temperature)
        resp = llm.send_prompt(system, user, max_tokens=500)
        if not resp.get("success"):
            return (f"ERROR: {resp.get('error')}", prompt)
        return (_clean(resp["response"]), prompt)


class SnartPromptEnhancer:
    """Stage 2: enhance a prompt using a chosen style / art direction."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "The basic idea"}),
                "style": ("STRING", {"multiline": True, "default": "", "placeholder": "Chosen style / art direction (from the Style Suggester, or your own words)"}),
                "extra_guidance": ("STRING", {"default": "", "placeholder": "Optional extra instruction, e.g. 'keep under 60 words'"}),
                **_llm_widgets(),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "status")
    FUNCTION = "enhance"
    CATEGORY = "Snart"

    def enhance(self, prompt, style, extra_guidance, llm_backend, api_endpoint,
                model_name, temperature) -> Tuple[str, str]:
        prompt = (prompt or "").strip()
        if not prompt:
            return ("", "ERROR: no prompt")

        style = (style or "").strip()
        extra = (extra_guidance or "").strip()
        extra_line = f"\nAdditional instruction: {extra}" if extra else ""

        system = (
            "You write vivid, detailed image-generation prompts. "
            "Output only the final prompt text - no preamble, no explanation, no markdown."
        )
        user = (
            f"Original idea:\n\"{prompt}\"\n\n"
            f"Style / art direction to apply:\n\"{style or 'a tasteful, well-balanced style'}\"\n\n"
            f"Rewrite the idea into a single rich, detailed image prompt that follows the style direction.{extra_line}"
        )

        llm = _client(llm_backend, api_endpoint, model_name, temperature)
        resp = llm.send_prompt(system, user, max_tokens=800)
        if not resp.get("success"):
            return ("", f"ERROR: {resp.get('error')}")
        return (_clean(resp["response"]), "OK")


NODE_CLASS_MAPPINGS = {
    "SnartStyleSuggester": SnartStyleSuggester,
    "SnartPromptEnhancer": SnartPromptEnhancer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SnartStyleSuggester": "Style Suggester",
    "SnartPromptEnhancer": "Prompt Enhancer",
}
