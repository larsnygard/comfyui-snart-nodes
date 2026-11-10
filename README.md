# Snart nodes for ComfyUI

Lars' own ComfyUI node pack. License: AGPL-3.0 (see `LICENSE`).

## Nodes

### LoRA Stacker (`LoraStacker`)

Stack multiple LoRAs onto a MODEL + CLIP in one node, with three modes:

- **manual** — paste `name.safetensors:0.8:0.8` lines (model/clip strength; a
  single `:0.8` applies to both). Names resolve case-insensitively and tolerate
  partial names.
- **random** — pick N random LoRAs. `match_checkpoint` restricts the pool to
  LoRAs whose base model matches the loaded checkpoint (SDXL / SD1.5 / Flux / …),
  so you don't mix architectures. Seeded via `random_seed` (0 = truly random).
- **llm** — ask an LLM to pick LoRAs from the installed list that best match a
  text description.

Outputs: `MODEL`, `CLIP`, `lora_stack` (for chaining), and `applied_loras` (a
readable summary of what was stacked).

**Chaining:** each stacker accepts a `lora_stack` input (from another stacker's
`lora_stack` output). Those LoRAs are applied first, as received, then this
node's own selection on top — so you can combine, e.g., a manual list with an
LLM/random pick. The format is the same 3-tuple `(name, model_strength,
clip_strength)` used by Efficiency nodes, so stacks interoperate.

### Empty Latent (`SnartEmptyLatent`)

Create an empty latent with four modes:

- **ratio** — 15 preset aspect ratios (1:1, 4:3, 3:2, 16:9, 21:9, and their
  portrait inverses), sized by a target megapixel count.
- **custom** — exact width/height.
- **auto** — guess the recommended size from the model's base architecture
  (SD1.5 → 512, SDXL/Flux/SD3 → 1024, …), applied to the chosen ratio.
- **random** — three sub-modes: `free` (independent random W/H in
  [min,max]), `preset` (random ratio, short side in [min,max]), `preset_mpix`
  (random ratio at a target megapixel count).

Dimensions are snapped to multiples of 8. Outputs `LATENT` plus the resolved
`width`/`height`.

### Prompt Enhancer — staged (`SnartStyleSuggester`, `SnartPromptEnhancer`)

Build a prompt in two LLM-driven steps, with no hardcoded style presets:

1. **Style Suggester** — give it a prompt, it asks the LLM for a list of
   distinct style / art directions to choose from.
2. **Prompt Enhancer** — give it the prompt plus the style you picked (from
   step 1 or your own words), it writes the final detailed prompt.

Works with any backend in `llm_backend`. API keys are read from `config.json`
or an environment variable only — there is no `api_key` widget, so keys cannot
leak into workflow or image metadata.

## LLM backends

The `llm` mode uses `llm_backend.py`, an OpenAI-compatible chat client. Local
servers (Strata, Kindi, or any OpenAI-compatible endpoint) and hosted providers
(OpenAI, DeepSeek, Mistral, Gemini, OpenRouter, Groq, Together) are supported.

API keys are read from `config.json` (gitignored — copy `config.json.example`
and fill in your keys) or an environment variable. Keys are never stored in node
widgets, so they can't leak into saved workflows or image metadata.
