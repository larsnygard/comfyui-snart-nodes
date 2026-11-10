"""
Snart LoRA Stacker node.

Stacks multiple LoRAs onto a MODEL+CLIP in a single node, instead of chaining
many LoraLoader nodes. Three selection modes:

  - manual : read "name:model_strength:clip_strength" lines (one per line)
  - random : pick N random LoRAs, optionally restricted to the checkpoint's
             base model (SDXL/SD1.5/Flux/...) so you don't mix architectures
  - llm    : ask an LLM (through this pack's llm_backend) to choose LoRAs from
             the installed list that best match a text description

Base-model matching is heuristic: it reads the checkpoint architecture from the
model object and each LoRA's base model from its safetensors metadata
("ss_base_model_version" / "modelspec.base_model" / "base_model"), falling back
to filename hints. LoRAs with no detectable base model are excluded in "match"
mode; if that leaves nothing, the node falls back to all LoRAs and says so.
"""

import os
import re
import json
import random
from typing import Dict, List, Optional, Tuple

import folder_paths
import comfy.sd
import comfy.utils

from .llm_backend import LLMBackend, get_backend_options, get_backend_tooltip


class LoraStacker:
    """Apply a stack of LoRAs to a model and clip in one node."""

    _CLASS_BASE = {
        "SDXL": "sdxl", "SDXLRefiner": "sdxl", "SDXL_instructpix2pix": "sdxl",
        "SD15": "sd1.5", "SD15_instructpix2pix": "sd1.5", "SD_X4Upscaler": "sd1.5",
        "SD21UNCLIP": "sd2.1",
        "SD3": "sd3",
        "Flux": "flux", "Flux2": "flux", "Flux3": "flux", "LongCatImage": "flux",
        "Lumina2": "lumina2", "MingImage": "lumina2", "ZImagePixelSpace": "lumina2",
        "HunyuanDiT": "hunyuan", "PixArt": "pixart", "AuraFlow": "auraflow",
        "StableCascade_C": "stable_cascade", "StableCascade_B": "stable_cascade",
        "HunyuanVideo": "hunyuan_video", "LTXV": "ltxv", "LTXAV": "ltxv",
        "GenmoMochi": "mochi", "SVD_img2vid": "svd",
    }

    # Priority-ordered (keyword, canonical) pairs for classifying a base model
    # string. First hit wins.
    _BASE_KEYWORDS = [
        ("flux", "flux"),
        ("sdxl", "sdxl"), ("sd_xl", "sdxl"), ("sd-xl", "sdxl"),
        ("pony", "sdxl"), ("illustrious", "sdxl"), ("noob", "sdxl"),
        ("sd3.5", "sd3"), ("sd35", "sd3"), ("sd3", "sd3"),
        ("sd2.1", "sd2.1"), ("sd2", "sd2.1"),
        ("sd1.5", "sd1.5"), ("sd15", "sd1.5"), ("sd_1.5", "sd1.5"),
        ("sd_v1", "sd1.5"), ("sdv1", "sd1.5"),
        ("wan", "wan"),
        ("hunyuan", "hunyuan"),
        ("lumina", "lumina2"),
        ("pixart", "pixart"), ("auraflow", "auraflow"), ("aura_flow", "auraflow"),
        ("stable_cascade", "stable_cascade"), ("cascade", "stable_cascade"),
        ("svd", "svd"), ("ltxv", "ltxv"), ("mochi", "mochi"),
    ]

    _LORA_EXTENSIONS = (".safetensors", ".ckpt", ".pt")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to apply LoRAs to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model to apply LoRAs to."}),
                "mode": (["manual", "random", "llm"], {"default": "manual"}),
                "loras_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "one per line: name.safetensors:0.8:0.8\nor just: name.safetensors:0.8 (strength for both)",
                }),
                "num_loras": ("INT", {"default": 2, "min": 1, "max": 12, "step": 1}),
                "strength_model": ("FLOAT", {"default": 0.8, "min": -2.0, "max": 2.0, "step": 0.05}),
                "strength_clip": ("FLOAT", {"default": 0.8, "min": -2.0, "max": 2.0, "step": 0.05}),
                "match_checkpoint": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "random/llm: only consider LoRAs whose base model matches the checkpoint.",
                }),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "llm_backend": (get_backend_options(), {"default": "kindi", "tooltip": get_backend_tooltip()}),
                "api_endpoint": ("STRING", {"default": "http://localhost:1234/v1", "multiline": False}),
                "model_name": ("STRING", {"default": "", "placeholder": "blank = auto-detect from the backend"}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "llm mode: what to select, e.g. 'a cinematic film style and a detailed character face'",
                }),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {"tooltip": "LoRAs from another stacker, applied before this node's own selection."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "LORA_STACK", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "lora_stack", "applied_loras")
    FUNCTION = "stack"
    CATEGORY = "Snart"

    # --- main entry -----------------------------------------------------

    def stack(self, model, clip, mode, loras_text, num_loras, strength_model,
              strength_clip, match_checkpoint, random_seed, llm_backend,
              api_endpoint, model_name, temperature, description, lora_stack=None):
        # LoRAs from an upstream stacker are applied first, as received.
        received = self._normalize_stack(lora_stack)

        own: List[Tuple[str, float, float]] = []
        error: Optional[str] = None
        try:
            own = self._select_loras(
                mode=mode,
                loras_text=loras_text,
                num_loras=num_loras,
                sm=strength_model,
                sc=strength_clip,
                match=match_checkpoint,
                seed=random_seed,
                backend=llm_backend,
                endpoint=api_endpoint,
                model_name=model_name,
                temperature=temperature,
                description=description,
                model=model,
            )
        except Exception as exc:
            error = str(exc)

        combined = received + own
        for name, sm, sc in combined:
            model, clip = self._apply_lora(model, clip, name, sm, sc)

        summary = " | ".join(f"{n}:{sm}:{sc}" for n, sm, sc in combined)
        if not summary:
            summary = "(no LoRAs applied)"
        if error:
            summary += f"\n⚠ {error}"
        return (model, clip, combined, summary)

    # --- selection ------------------------------------------------------

    def _select_loras(self, mode, loras_text, num_loras, sm, sc, match, seed,
                      backend, endpoint, model_name, temperature, description, model):
        mode = (mode or "manual").strip().lower()

        if mode == "manual":
            return self._parse_manual(loras_text, sm, sc)
        if mode == "random":
            return self._random_select(num_loras, sm, sc, match, seed, model)
        if mode == "llm":
            return self._llm_select(num_loras, sm, sc, match, temperature, backend,
                                    endpoint, model_name, description, model)
        raise RuntimeError(f"Unknown mode: {mode}")

    def _parse_manual(self, text, default_sm, default_sc):
        result: List[Tuple[str, float, float]] = []
        missing: List[str] = []
        for line in (text or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(":")]
            name = parts[0]
            if not name:
                continue
            sm, sc = default_sm, default_sc
            if len(parts) >= 2 and parts[1]:
                try:
                    sm = float(parts[1])
                except ValueError:
                    pass
            if len(parts) >= 3 and parts[2]:
                try:
                    sc = float(parts[2])
                except ValueError:
                    pass
            resolved = self._resolve_lora_name(name)
            if resolved:
                result.append((resolved, sm, sc))
            else:
                missing.append(name)
        if missing:
            raise RuntimeError("LoRA(s) not found in models/loras: " + ", ".join(missing))
        return result

    def _random_select(self, num, sm, sc, match, seed, model):
        names = self._candidate_loras(match, model)
        if not names:
            raise RuntimeError("No LoRAs available in models/loras")
        rng = random.Random(seed) if seed else random.Random()
        picked = rng.sample(names, min(num, len(names)))
        return [(n, sm, sc) for n in picked]

    def _llm_select(self, num, sm, sc, match, temperature, backend, endpoint,
                    model_name, description, model):
        names = self._candidate_loras(match, model)
        if not names:
            raise RuntimeError("No LoRAs available in models/loras")
        names = names[:400]  # keep the LLM prompt bounded

        listing = "\n".join(f"- {n}" for n in names)
        system = (
            "You choose LoRAs for an image-generation workflow. "
            "Reply with ONLY a JSON object, no prose, no markdown fences."
        )
        user = (
            "Available LoRAs (use exact filenames only):\n"
            f"{listing}\n\n"
            f"Choose up to {num} LoRAs that best match this description: "
            f"\"{description or 'pick a diverse, interesting combination'}\".\n\n"
            "Reply with exactly:\n"
            '{"loras": [{"name": "<exact filename>", "strength": <0.2..1.0>}, ...]}\n'
            "Return {\"loras\": []} if nothing fits."
        )

        llm = LLMBackend(
            backend_type=backend,
            endpoint=endpoint,
            model_name=(model_name or "").strip() or None,
            temperature=temperature,
            api_key=None,
        )
        resp = llm.send_prompt(system, user, max_tokens=600)
        if not resp.get("success"):
            raise RuntimeError(f"LLM selection failed: {resp.get('error')}")

        data = self._parse_json(resp["response"])
        items = data.get("loras") or []
        result: List[Tuple[str, float, float]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = self._resolve_lora_name(str(item.get("name", "")))
            if not name:
                continue
            try:
                s = float(item.get("strength", sm))
            except (TypeError, ValueError):
                s = sm
            s = max(-2.0, min(2.0, s))
            result.append((name, s, s))
        if not result:
            raise RuntimeError("LLM returned no valid LoRA names")
        return result[:num]

    # --- lora discovery + base-model matching ---------------------------

    def _available_loras(self) -> List[str]:
        return [
            n for n in folder_paths.get_filename_list("loras")
            if n.lower().endswith(self._LORA_EXTENSIONS)
        ]

    def _candidate_loras(self, match, model) -> List[str]:
        names = self._available_loras()
        if not match:
            return names
        ckpt_base = self._checkpoint_base(model)
        matching = [n for n in names if self._lora_matches(n, ckpt_base)]
        if matching:
            return matching
        # Nothing matched the checkpoint -> fall back to everything rather than
        # applying nothing.
        return names

    def _resolve_lora_name(self, name: str) -> Optional[str]:
        name = (name or "").strip()
        if not name:
            return None
        loras = self._available_loras()
        lowered = name.lower()
        for n in loras:
            if n.lower() == lowered:
                return n
        stem = os.path.splitext(name)[0].lower()
        for n in loras:
            if os.path.splitext(n)[0].lower() == stem:
                return n
        # unique stem-substring match (forgiving partial names like "ColoringBookRedmond")
        hits = [n for n in loras if stem and stem in os.path.splitext(n)[0].lower()]
        if len(hits) == 1:
            return hits[0]
        # unique full-name substring match
        hits = [n for n in loras if lowered in n.lower()]
        return hits[0] if len(hits) == 1 else None

    def _checkpoint_base(self, model) -> Optional[str]:
        inner = getattr(model, "model", None)
        if inner is None:
            return None
        for cls in type(inner).__mro__:
            if cls.__name__ in self._CLASS_BASE:
                return self._CLASS_BASE[cls.__name__]
        return None

    def _lora_matches(self, name: str, ckpt_base: Optional[str]) -> bool:
        if ckpt_base is None:
            return True  # couldn't detect the checkpoint -> don't filter
        lora_base = self._lora_base_model(name)
        if lora_base is None:
            return False  # unknown LoRA -> exclude in strict match
        return lora_base == ckpt_base

    def _lora_base_model(self, name: str) -> Optional[str]:
        path = folder_paths.get_full_path("loras", name)
        if not path:
            return None
        meta = self._read_lora_metadata(path)
        raw = " ".join(str(meta.get(k, "")) for k in (
            "ss_base_model_version", "modelspec.base_model", "base_model",
            "ss_sd_model_name", "modelspec.architecture",
        ))
        if not raw.strip():
            raw = name
        return self._classify_base(raw)

    def _classify_base(self, raw: str) -> Optional[str]:
        raw = (raw or "").lower()
        for keyword, canonical in self._BASE_KEYWORDS:
            if keyword in raw:
                return canonical
        return None

    @staticmethod
    def _read_lora_metadata(path: str) -> Dict:
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt") as handle:
                return dict(handle.metadata() or {})
        except Exception:
            pass
        try:
            header = comfy.utils.safetensors_header(path)
            if header:
                parsed = json.loads(header)
                return dict((parsed.get("__metadata__") or {}))
        except Exception:
            pass
        return {}

    # --- incoming LORA_STACK --------------------------------------------

    @staticmethod
    def _normalize_stack(stack):
        """Normalize an incoming LORA_STACK to a list of (name, model_strength, clip_strength).

        Accepts the 3-tuple form this node and Efficiency nodes emit, or bare
        strings (default strength 1.0). Skips "None"/empty entries.
        """
        result = []
        if not stack:
            return result
        for entry in stack:
            if isinstance(entry, str):
                name, sm, sc = entry, 1.0, 1.0
            elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                name, sm, sc = entry[0], entry[1], entry[2]
            else:
                continue
            name = str(name).strip()
            if not name or name.lower() == "none":
                continue
            try:
                sm = float(sm)
                sc = float(sc)
            except (TypeError, ValueError):
                sm, sc = 1.0, 1.0
            result.append((name, sm, sc))
        return result

    # --- applying -------------------------------------------------------

    @staticmethod
    def _apply_lora(model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return model, clip
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora, lora_metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip, lora_metadata=lora_metadata
        )
        return model_lora, clip_lora

    # --- json parsing ---------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Dict:
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


NODE_CLASS_MAPPINGS = {"LoraStacker": LoraStacker}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraStacker": "LoRA Stacker"}
