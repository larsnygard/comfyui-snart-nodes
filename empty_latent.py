"""
Snart Empty Latent node.

Creates an empty latent (like ComfyUI's EmptyLatentImage) with more control:

  - ratio  : pick a preset aspect ratio, sized by a target megapixel count
  - custom : exact width/height
  - auto   : guess the recommended size from the model's base architecture
             (SD1.5 -> 512, SDXL/Flux/SD3 -> 1024, ...), applied to the chosen ratio
  - random : three sub-modes:
        free        - totally random width AND height, each in [min_size, max_size]
        preset      - random ratio, short side in [min_size, max_size]
        preset_mpix - random ratio, sized to a target megapixel count

Dimensions are snapped to multiples of 8 (the latent is 1/8 resolution) and
clamped to >= 64.
"""

import math
import random

import torch
import comfy.model_management


class SnartEmptyLatent:

    # (label, ratio_w, ratio_h)
    _RATIOS = [
        ("1:1", 1, 1),
        ("5:4", 5, 4),
        ("4:3", 4, 3),
        ("3:2", 3, 2),
        ("16:10", 16, 10),
        ("16:9", 16, 9),
        ("21:9", 21, 9),
        ("2:1", 2, 1),
        ("4:5", 4, 5),
        ("3:4", 3, 4),
        ("2:3", 2, 3),
        ("10:16", 10, 16),
        ("9:16", 9, 16),
        ("9:21", 9, 21),
        ("1:2", 1, 2),
    ]
    _RATIO_BY_LABEL = {label: (w, h) for label, w, h in _RATIOS}
    _RATIO_PAIRS = [(w, h) for _, w, h in _RATIOS]

    # model base architecture -> recommended target megapixels (auto mode)
    _BASE_MPIX = {
        "sd1.5": 0.26,          # 512x512
        "sd2.1": 0.59,          # 768x768
        "sdxl": 1.05,           # 1024x1024
        "sd3": 1.05,
        "flux": 1.05,
        "hunyuan": 1.05,
        "pixart": 1.05,
        "auraflow": 1.05,
        "lumina2": 1.05,
        "stable_cascade": 1.05,
        "wan": 1.05,
        "hunyuan_video": 1.05,
        "svd": 1.05,
        "mochi": 1.05,
    }
    _DEFAULT_MPIX = 1.0

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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["ratio", "custom", "auto", "random"], {"default": "ratio"}),
                "ratio": ([r[0] for r in cls._RATIOS], {"default": "1:1"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "random_mode": (["free", "preset", "preset_mpix"], {"default": "preset"}),
                "min_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "max_size": ("INT", {"default": 1536, "min": 64, "max": 8192, "step": 8}),
                "target_mpix": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "Snart"

    def generate(self, mode, ratio, megapixels, batch_size, width, height,
                 random_mode, min_size, max_size, target_mpix, seed, model=None):
        width, height = self._resolve(
            mode=mode,
            ratio=ratio,
            megapixels=megapixels,
            width=width,
            height=height,
            random_mode=random_mode,
            min_size=min_size,
            max_size=max_size,
            target_mpix=target_mpix,
            seed=seed,
            model=model,
        )
        latent = torch.zeros(
            [batch_size, 4, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        return ({"samples": latent, "downscale_ratio_spacial": 8}, width, height)

    # --- resolution -----------------------------------------------------

    def _resolve(self, mode, ratio, megapixels, width, height, random_mode,
                 min_size, max_size, target_mpix, seed, model):
        mode = (mode or "ratio").strip().lower()

        if mode == "custom":
            return self._snap8(width), self._snap8(height)

        if mode == "ratio":
            return self._dims_from_ratio(ratio, megapixels)

        if mode == "auto":
            mpix = self._auto_mpix(model)
            return self._dims_from_ratio(ratio, mpix)

        if mode == "random":
            rng = random.Random(seed) if seed else random.Random()
            if random_mode == "free":
                return self._snap8(rng.randint(min_size, max_size)), self._snap8(rng.randint(min_size, max_size))
            if random_mode == "preset":
                rw, rh = rng.choice(self._RATIO_PAIRS)
                short = rng.randint(min_size, max_size)
                return self._dims_from_short(short, rw, rh)
            if random_mode == "preset_mpix":
                rw, rh = rng.choice(self._RATIO_PAIRS)
                return self._dims_from_pair(target_mpix, rw, rh)

        raise RuntimeError(f"Unknown mode: {mode}")

    def _dims_from_ratio(self, label, mpix):
        rw, rh = self._RATIO_BY_LABEL.get(label, (1, 1))
        return self._dims_from_pair(mpix, rw, rh)

    @staticmethod
    def _dims_from_pair(mpix, rw, rh):
        total = float(mpix) * 1_000_000.0
        w = math.sqrt(total * rw / rh)
        h = total / w
        return SnartEmptyLatent._snap8(round(w)), SnartEmptyLatent._snap8(round(h))

    @staticmethod
    def _dims_from_short(short, rw, rh):
        short = int(short)
        if rw >= rh:
            w = short * rw / rh
            h = short
        else:
            w = short
            h = short * rh / rw
        return SnartEmptyLatent._snap8(round(w)), SnartEmptyLatent._snap8(round(h))

    @staticmethod
    def _snap8(value):
        return max(64, int(round(int(value) / 8.0)) * 8)

    # --- auto (model recommendation) ------------------------------------

    def _auto_mpix(self, model):
        base = self._checkpoint_base(model)
        if base is None:
            return self._DEFAULT_MPIX
        return self._BASE_MPIX.get(base, self._DEFAULT_MPIX)

    def _checkpoint_base(self, model):
        inner = getattr(model, "model", None)
        if inner is None:
            return None
        for cls in type(inner).__mro__:
            if cls.__name__ in self._CLASS_BASE:
                return self._CLASS_BASE[cls.__name__]
        return None


NODE_CLASS_MAPPINGS = {"SnartEmptyLatent": SnartEmptyLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"SnartEmptyLatent": "Empty Latent"}
