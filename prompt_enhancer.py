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

import random as _random
import re
from typing import List, Optional, Tuple

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
        "Reply with a plain list only - no numbering, no commentary."
    )
    user = (
        f"Image idea: \"{prompt}\"\n"
        f"Chosen so far: {chosen_txt}\n\n"
        f"Refinement step {step}. Propose {num_suggestions} options for the next refinement.\n"
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
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_prompt", "log")
    FUNCTION = "run"
    CATEGORY = "Snart"

    def run(self, prompt, num_suggestions, num_steps, broadness, random, random_seed,
            choices, llm_backend, api_endpoint, model_name, temperature) -> Tuple[str, str]:
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


NODE_CLASS_MAPPINGS = {"SnartStagedPrompter": SnartStagedPrompter}
NODE_DISPLAY_NAME_MAPPINGS = {"SnartStagedPrompter": "Staged Prompt Builder"}
