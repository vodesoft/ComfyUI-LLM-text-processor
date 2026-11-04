from __future__ import annotations

from .folder_registry import (
    NO_MMPROJ,
    full_mmproj_path,
    full_model_path,
    full_system_prompt_path,
    mmproj_options,
    model_options,
    system_prompt_options,
)
from .llama_cli import (
    MAX_LLAMA_SEED,
    build_command,
    run_llama_cli,
    split_extra_args,
)
from comfy_api.latest import ComfyExtension, io


class LLMTextProcessor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LLMTextProcessor",
            display_name="LLM Text Processor",
            description="Run a GGUF LLM (optionally with image media) over a prompt.",
            category="advanced/LLM Text Processor",
            inputs=[
                io.Combo.Input("model", options=model_options(),
                               tooltip="GGUF model loaded from ComfyUI/models/LLM. mmproj files are hidden from this list."),
                io.Combo.Input("mmproj", options=mmproj_options(), default=NO_MMPROJ,
                               tooltip="Vision projector GGUF. Required when one or more images are connected."),
                io.Combo.Input("system_prompt", options=system_prompt_options(), default=NO_MMPROJ,
                               tooltip="System prompt preset from ComfyUI/models/LLM/prompts, or none."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                default="Describe this image in detail.",
                                tooltip="User prompt sent to the selected model."),
                io.Int.Input("max_tokens", default=2048, min=1, max=32768,
                             tooltip="Maximum number of tokens to generate."),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05,
                               tooltip="Sampling temperature. Lower is more deterministic."),
                io.Float.Input("top_p", default=0.8, min=0.0, max=1.0, step=0.01,
                               tooltip="Nucleus sampling threshold."),
                io.Int.Input("top_k", default=20, min=1, max=1000,
                             tooltip="Top-K sampling cutoff."),
                io.Float.Input("repeat_penalty", default=1.0, min=0.0, max=3.0, step=0.01,
                               tooltip="Penalty applied to repeated tokens."),
                io.Int.Input("ctx_size", default=8192, min=512, max=1048576, step=512,
                             tooltip="Context window size in tokens. Use a value supported by the selected GGUF; larger context uses more VRAM."),
                io.Combo.Input("memory_mode", options=["auto", "gpu_layers", "cpu_moe_layers", "gpu_and_cpu_moe_layers"],
                               default="auto", advanced=True,
                               tooltip="Advanced memory placement mode: auto, gpu_layers, cpu_moe_layers, or gpu_and_cpu_moe_layers."),
                io.Int.Input("n_gpu_layers", default=99, min=-1, max=999, advanced=True,
                             tooltip="Used only in gpu_layers and gpu_and_cpu_moe_layers modes. Number of model layers to place on the GPU."),
                io.Int.Input("n_cpu_moe_layers", default=1, min=1, max=999, advanced=True,
                             tooltip="Used only in cpu_moe_layers and gpu_and_cpu_moe_layers modes. Number of MoE layers to keep on the CPU."),
                io.Int.Input("seed", default=1, min=-1, max=MAX_LLAMA_SEED,
                             tooltip="Random seed. Use -1 for a random seed."),
                io.Int.Input("timeout_seconds", default=300, min=10, max=3600, advanced=True,
                             tooltip="Maximum time to wait before generation is stopped."),
                io.Combo.Input("reasoning", options=["auto", "on", "off"], default="off", advanced=True,
                               tooltip="Reasoning output mode."),
                io.Image.Input("image1", optional=True,
                               tooltip="Optional first image input. A single image or ComfyUI batch is passed to llama.cpp."),
                io.Image.Input("image2", optional=True,
                               tooltip="Optional second image input. Used only when connected."),
                io.Image.Input("image3", optional=True,
                               tooltip="Optional third image input. Used only when connected."),
                io.Image.Input("image4", optional=True,
                               tooltip="Optional fourth image input. Used only when connected."),
                io.Boolean.Input("enable_processing", default=True, advanced=True,
                                 tooltip="When enabled, run normal node processing. When disabled, forward the input prompt directly as RESPONSE."),
                io.String.Input("extra_args", multiline=False, default="", advanced=True,
                                tooltip="Optional advanced llama.cpp parameters. Leave empty for normal use."),
            ],
            outputs=[
                io.String.Output(display_name="RESPONSE", tooltip="Final model response with reasoning blocks removed."),
                io.String.Output(display_name="REASONING", tooltip="Extracted reasoning when present in model output."),
                io.String.Output(display_name="PERF", tooltip="llama.cpp prompt and generation speed."),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        model,
        mmproj,
        system_prompt,
    ):
        return True

    @classmethod
    def execute(
        cls,
        model: str,
        mmproj: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        ctx_size: int,
        memory_mode: str,
        n_gpu_layers: int,
        n_cpu_moe_layers: int,
        seed: int,
        timeout_seconds: int,
        reasoning: str,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        enable_processing: bool = True,
        extra_args: str = "",
    ):
        if not enable_processing:
            return io.NodeOutput(prompt, "", "")

        current_models = model_options()
        current_mmprojs = mmproj_options()
        current_system_prompts = system_prompt_options()

        if model not in current_models:
            raise ValueError(f"Model not found: {model}")

        if mmproj not in current_mmprojs:
            raise ValueError(f"mmproj not found: {mmproj}")

        if system_prompt not in current_system_prompts:
            raise ValueError(f"System prompt preset not found: {system_prompt}")

        model_path = full_model_path(model)
        mmproj_path = full_mmproj_path(mmproj)
        system_prompt_path = full_system_prompt_path(system_prompt)
        parsed_extra_args = split_extra_args(extra_args)

        command, cleanup_paths = build_command(
            model_path=model_path,
            mmproj_path=mmproj_path,
            system_prompt_path=system_prompt_path,
            images=(image1, image2, image3, image4),
            audios=None,
            videos=None,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            ctx_size=ctx_size,
            memory_mode=memory_mode,
            n_gpu_layers=n_gpu_layers,
            n_cpu_moe_layers=n_cpu_moe_layers,
            seed=seed,
            reasoning=reasoning,
            extra_args=parsed_extra_args,
        )
        response, reasoning_text, perf = run_llama_cli(
            command=command,
            timeout_seconds=timeout_seconds,
            cleanup_paths=cleanup_paths,
        )
        return io.NodeOutput(response, reasoning_text, perf)


