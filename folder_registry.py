from __future__ import annotations

import os
from pathlib import Path

import folder_paths


LLM_FOLDER = "llm_text_processor_models"
PROMPT_FOLDER = "llm_text_processor_prompts"
NO_SYSTEM_PROMPT = "none"
NO_MMPROJ = "none"
NO_MODELS_FOUND = "No GGUF models found"


# Standard model-type keys. ComfyUI (and the Desktop "Storage -> Models"
# settings) register every model root -- the main folder and any shared
# directories -- under these keys. The parent of each registered path is a
# model root, so we scan the LLM subfolder inside all of them.
STANDARD_MODEL_KEYS = (
    "checkpoints",
    "configs",
    "loras",
    "vae",
    "vae_approx",
    "text_encoders",
    "diffusion_models",
    "clip_vision",
    "style_models",
    "embeddings",
    "controlnet",
    "gligen",
    "upscale_models",
    "latent_upscale_models",
    "hypernetworks",
    "photomaker",
    "classifiers",
    "model_patches",
    "audio_encoders",
    "background_removal",
    "frame_interpolation",
    "geometry_estimation",
    "optical_flow",
    "detection",
)


def llm_root() -> Path:
    return Path(folder_paths.models_dir) / "LLM"


def prompt_root() -> Path:
    return llm_root() / "prompts"


def model_roots() -> list[Path]:
    roots: list[Path] = [Path(folder_paths.models_dir)]
    for key in STANDARD_MODEL_KEYS:
        for path in folder_paths.get_folder_paths(key):
            root = Path(path).parent
            if root not in roots:
                roots.append(root)
    return roots


def register_folders() -> None:
    llm_root().mkdir(parents=True, exist_ok=True)
    prompt_root().mkdir(parents=True, exist_ok=True)

    # ComfyUI's add_model_folder_path cannot set extensions for a new key, so we
    # register the custom keys directly with the same tuple shape ComfyUI uses.
    # The LLM subfolder is registered for every model root: the main folder first,
    # then any shared directories added through the Desktop "Storage -> Models"
    # settings or extra_model_paths.yaml. The main folder stays the primary source.
    _set_folder_paths(LLM_FOLDER, [str(root / "LLM") for root in model_roots()], {".gguf"})
    _set_folder_paths(
        PROMPT_FOLDER,
        [str(root / "LLM" / "prompts") for root in model_roots()],
        {".txt", ".md"},
    )


def _set_folder_paths(key: str, ordered_paths: list[str], extensions: set[str]) -> None:
    entry = folder_paths.folder_names_and_paths.get(key)
    existing = entry[0] if entry else []
    merged = list(ordered_paths)
    for path in existing:
        if path not in merged:
            merged.append(path)
    folder_paths.folder_names_and_paths[key] = (merged, extensions)


def model_options() -> list[str]:
    files = folder_paths.get_filename_list(LLM_FOLDER)
    models = [name for name in files if "mmproj" not in Path(name).name.lower()]
    return models or [NO_MODELS_FOUND]


def mmproj_options() -> list[str]:
    files = folder_paths.get_filename_list(LLM_FOLDER)
    mmproj = [name for name in files if "mmproj" in Path(name).name.lower()]
    return [NO_MMPROJ] + mmproj


def system_prompt_options() -> list[str]:
    files = folder_paths.get_filename_list(PROMPT_FOLDER)
    top_level_files = [name for name in files if os.sep not in name and "/" not in name]
    return [NO_SYSTEM_PROMPT] + top_level_files


def full_model_path(name: str) -> Path:
    if name == NO_MODELS_FOUND:
        raise FileNotFoundError(
            f"No GGUF model files were found in {llm_root()}. Place a .gguf model there and refresh ComfyUI."
        )
    path = folder_paths.get_full_path(LLM_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"GGUF model not found: {name}")
    return Path(path)


def full_mmproj_path(name: str) -> Path | None:
    if name == NO_MMPROJ:
        return None
    path = folder_paths.get_full_path(LLM_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"mmproj GGUF file not found: {name}")
    return Path(path)


def full_system_prompt_path(name: str) -> Path | None:
    if name == NO_SYSTEM_PROMPT:
        return None
    path = folder_paths.get_full_path(PROMPT_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"System prompt preset not found: {name}")
    return Path(path)


register_folders()
