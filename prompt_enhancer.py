"""Snart staged prompt builder.

One node that refines an image prompt in steps with an LLM, then writes the
final prompt. No hardcoded presets - at each step the LLM proposes options, and
either you pick one (from a dropdown, via the "Suggest" button) or the node
picks at random.

  - num_suggestions : how many options the LLM proposes per step
  - num_steps       : how many refinement steps before the final prompt
  - broadness       : how terse or evocative each option is (simple/specific/broad)
  - random          : tick to auto-pick a random option each step and continue
  - choices         : your manual picks, one per line (filled by the dropdown, or typed)

API keys are read from config.json / environment variables only - no api_key
widget, so keys cannot leak.
"""

import json
import os
import random as _random
import re
from typing import List, Optional, Tuple

import folder_paths

from .llm_backend import LLMBackend, get_backend_options, get_backend_tooltip

_BROADNESS = {
    "simple": "Give very short options, 1-3 words each (e.g. 'photograph', 'oil painting').",
    "specific": "Give specific, concrete style names (e.g. 'lomo photography', 'watercolor (aquarelle)').",
    "broad": "Give rich, evocative one-line descriptions (e.g. 'a stunning color photograph in National Geographic style', 'a colorful watercolor painting done by a four-year-old girl').",
}


def _client(llm_backend, api_endpoint, model_name, temperature):
    return LLMBackend(
        backend_type=llm_backend,
        endpoint=api_endpoint,
        model_name=(model_name or "").strip() or None,
        temperature=temperature,
        api_key=None,
    )


def _clean(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


def _parse_list(text: str, limit: Optional[int] = None) -> List[str]:
    """Turn an LLM list reply into clean option strings (strip numbering/bullets)."""
    items: List[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:\d+[.)\-:]+|[-*•])\s*", "", line).strip()
        if line and line not in items:
            items.append(line)
        if limit and len(items) >= limit:
            break
    return items


def suggest_options(llm: LLMBackend, prompt: str, chosen: List[str], step: int,
                    num_suggestions: int, broadness: str) -> List[str]:
    """Ask the LLM for the next step's options. Returns a list of strings."""
    detail = _BROADNESS.get(broadness, _BROADNESS["simple"])
    chosen_txt = ", ".join(chosen) if chosen else "(nothing yet)"
    system = (
        "You are refining an image prompt step by step. "
        "Each step builds on the choices already made: every option must be "
        "consistent with, and compatible with, everything chosen so far. "
        "Never contradict an earlier choice (e.g. no 'photograph' after "
        "'oil painting', no subject that replaces the previous one). "
        "Reply with a plain list only - no numbering, no commentary."
    )
    user = (
        f"Image idea: \"{prompt}\"\n"
        f"Chosen so far (all of these must remain true together): {chosen_txt}\n\n"
        f"Refinement step {step}: propose {num_suggestions} options that build on "
        f"and stay consistent with the choices above.\n"
        f"{detail}\n\n"
        f"Reply with exactly {num_suggestions} options, one per line."
    )
    resp = llm.send_prompt(system, user, max_tokens=400)
    if not resp.get("success"):
        return []
    return _parse_list(resp.get("response", ""), num_suggestions)


def finalize_prompt(llm: LLMBackend, prompt: str, chosen: List[str]) -> str:
    """Ask the LLM for the final enhanced prompt combining everything."""
    system = (
        "You write vivid, detailed image-generation prompts. "
        "Output only the final prompt text - no preamble, no explanation, no markdown."
    )
    user = (
        f"Original idea: \"{prompt}\"\n"
        f"Chosen refinements: {', '.join(chosen)}\n\n"
        f"Write a single rich, detailed image prompt that combines the idea with these choices."
    )
    resp = llm.send_prompt(system, user, max_tokens=800)
    if not resp.get("success"):
        return f"ERROR: {resp.get('error')}"
    return _clean(resp.get("response", ""))


class SnartStagedPrompter:
    """Staged prompt refinement: LLM proposes options per step, then writes the final prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "The basic idea, e.g. 'a fox'"}),
                "num_suggestions": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "num_steps": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                "broadness": (["simple", "specific", "broad"], {"default": "simple"}),
                "random": ("BOOLEAN", {"default": False, "tooltip": "Auto-pick a random option each step and continue."}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "choices": ("STRING", {"multiline": True, "default": "", "placeholder": "Your picks, one per line (filled by the Suggest button, or typed)."}),
                "llm_backend": (get_backend_options(), {"default": "kindi", "tooltip": get_backend_tooltip()}),
                "api_endpoint": ("STRING", {"default": "http://localhost:1234/v1"}),
                "model_name": ("STRING", {"default": "", "placeholder": "blank = auto-detect"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "choice": ([""], {"default": "", "tooltip": "Pick an option here, then click Suggest again to add it and get the next step's options."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_prompt", "log")
    FUNCTION = "run"
    CATEGORY = "Snart"

    def run(self, prompt, num_suggestions, num_steps, broadness, random, random_seed,
            choices, llm_backend, api_endpoint, model_name, temperature, choice=None) -> Tuple[str, str]:
        prompt = (prompt or "").strip()
        if not prompt:
            return ("", "ERROR: no prompt")

        picks = [c for c in (choices or "").splitlines() if c.strip()]
        llm = _client(llm_backend, api_endpoint, model_name, temperature)
        rng = _random.Random(random_seed) if random_seed else _random.Random()

        chosen: List[str] = []
        log = [f"Prompt: {prompt}"]

        for step in range(1, int(num_steps) + 1):
            opts = suggest_options(llm, prompt, chosen, step, int(num_suggestions), broadness)
            if not opts:
                log.append(f"\nERROR at step {step}: the LLM returned no options.")
                return ("", "\n".join(log))

            if random:
                pick = rng.choice(opts)
            else:
                pick = picks[step - 1] if step - 1 < len(picks) else opts[0]
            chosen.append(pick)

            log.append(f"\nStep {step} ({broadness}):")
            for i, opt in enumerate(opts, 1):
                log.append(f"  {'>>' if opt == pick else '  '} {i}. {opt}")

        final = finalize_prompt(llm, prompt, chosen)
        log.append(f"\nChosen: {' / '.join(chosen)}")
        log.append(f"\nFinal prompt:\n{final}")
        return (final, "\n".join(log))


_PLATFORMS = ["flux", "sd_xl", "sd_1_5", "pony", "illustrious", "noobai", "qwen_image", "wan_image", "hunyuan", "pixart", "chroma"]

_PLATFORM_GUIDANCE = {
    "flux": "Flux understands natural language - write flowing descriptive sentences, avoid tag spam.",
    "sd_xl": "SDXL works best with comma-separated tags; use terse tags and (emphasis:1.2) weights where useful.",
    "sd_1_5": "SD1.5 uses comma-separated tags; include quality tags like 'masterpiece, best quality'.",
    "pony": "Pony uses tags plus score tags; start with 'score_9, score_8_up, score_7_up'.",
    "illustrious": "Illustrious uses tags; include rating/quality tags.",
    "noobai": "NoobAI uses tags; include quality tags.",
    "qwen_image": "Qwen-Image understands natural language - write descriptive prose.",
    "wan_image": "Wan understands natural language - write descriptive prose.",
    "hunyuan": "HunyuanDiT understands natural language.",
    "pixart": "PixArt understands natural language.",
    "chroma": "Chroma understands natural language.",
}


class SnartPromptBuilder:
    """Expanding staged selector: each step's pick populates the next dropdown.

    Click 🎲 Start to fetch step 1, then pick — the next step's dropdown appears
    automatically. Outputs the accumulated text as-is, for an enhancer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "The basic idea, e.g. 'a fox'"}),
                "num_steps": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                "num_suggestions": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "broadness": (["simple", "specific", "broad"], {"default": "simple"}),
                "choices": ("STRING", {"multiline": True, "default": "", "placeholder": "Auto-filled as you pick each step."}),
                "llm_backend": (get_backend_options(), {"default": "kindi", "tooltip": get_backend_tooltip()}),
                "api_endpoint": ("STRING", {"default": "http://localhost:1234/v1"}),
                "model_name": ("STRING", {"default": "", "placeholder": "blank = auto-detect"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "build"
    CATEGORY = "Snart"

    def build(self, prompt, num_steps, num_suggestions, broadness, choices,
              llm_backend, api_endpoint, model_name, temperature):
        prompt = (prompt or "").strip()
        picks = [c for c in (choices or "").splitlines() if c.strip()]
        if not prompt and not picks:
            return ("",)
        if prompt and picks:
            return (f"{prompt}. {', '.join(picks)}.",)
        return (prompt or ", ".join(picks),)


_NEGATIVE_PLATFORMS = {"sd_xl", "sd_1_5", "pony", "illustrious", "noobai"}


def _parse_json(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise RuntimeError("Could not parse JSON from the LLM response")


def _available_embeddings(limit: int = 40) -> List[str]:
    """Installed textual-inversion embedding names (without extension)."""
    try:
        names = folder_paths.get_filename_list("embeddings")
    except Exception:
        return []
    result = []
    for n in names:
        stem = os.path.splitext(os.path.basename(str(n)))[0]
        if stem and stem not in result:
            result.append(stem)
        if len(result) >= limit:
            break
    return result


class SnartPromptEnhancer:
    """Fine-tune a prompt for a target platform; also gives a negative prompt
    (where the platform uses one) and can reference installed embeddings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Text to enhance (e.g. from the Prompt Builder)."}),
                "platform": (_PLATFORMS, {"default": "flux"}),
                "extra_guidance": ("STRING", {"default": "", "placeholder": "Optional extra instruction, e.g. 'keep under 60 words'"}),
                "embeddings": ("STRING", {"default": "", "placeholder": "Optional: embedding names to include (comma separated), or blank to let the LLM pick."}),
                "llm_backend": (get_backend_options(), {"default": "kindi", "tooltip": get_backend_tooltip()}),
                "api_endpoint": ("STRING", {"default": "http://localhost:1234/v1"}),
                "model_name": ("STRING", {"default": "", "placeholder": "blank = auto-detect"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "negative_prompt", "status")
    FUNCTION = "enhance"
    CATEGORY = "Snart"

    def enhance(self, text, platform, extra_guidance, embeddings, llm_backend,
                api_endpoint, model_name, temperature):
        text = (text or "").strip()
        if not text:
            return ("", "", "ERROR: no text")
        guidance = _PLATFORM_GUIDANCE.get(platform, "Write a detailed image prompt.")
        extra = (extra_guidance or "").strip()
        extra_line = f"\nAdditional instruction: {extra}" if extra else ""

        wants_negative = platform in _NEGATIVE_PLATFORMS
        available = _available_embeddings()
        forced = [e for e in (embeddings or "").replace(",", "\n").splitlines() if e.strip()]

        emb_line = ""
        if available:
            emb_line = f"\nInstalled embeddings (reference by name if useful): {', '.join(available)}."
        if forced:
            emb_line += f"\nYou MUST include these embeddings in the prompt: {', '.join(forced)}."

        system = (
            "You write image-generation prompts. Reply with ONLY a JSON object "
            'of the form {"positive": "...", "negative": "..."}. No prose, no markdown.'
        )
        user = (
            f"Text to enhance:\n\"{text}\"\n\n"
            f"Target platform: {platform}.\n{guidance}{extra_line}{emb_line}\n\n"
            f"Write the positive prompt optimized for {platform}."
        )
        if wants_negative:
            user += "\nAlso write a negative prompt (things to avoid: low quality, bad anatomy, artifacts, blur, duplicates, etc.)."
        else:
            user += f"\n{platform} does not use negative prompts - set \"negative\" to an empty string."

        llm = _client(llm_backend, api_endpoint, model_name, temperature)
        resp = llm.send_prompt(system, user, max_tokens=900)
        if not resp.get("success"):
            return ("", "", f"ERROR: {resp.get('error')}")
        try:
            data = _parse_json(resp.get("response", ""))
            positive = str(data.get("positive") or "").strip()
            negative = str(data.get("negative") or "").strip()
        except Exception:
            positive = _clean(resp.get("response", ""))
            negative = ""
        if not wants_negative:
            negative = ""
        return (positive, negative, "OK")


NODE_CLASS_MAPPINGS = {
    "SnartStagedPrompter": SnartStagedPrompter,
    "SnartPromptBuilder": SnartPromptBuilder,
    "SnartPromptEnhancer": SnartPromptEnhancer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SnartStagedPrompter": "Staged Prompt Builder",
    "SnartPromptBuilder": "Prompt Builder",
    "SnartPromptEnhancer": "Prompt Enhancer",
}
