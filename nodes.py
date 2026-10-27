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


class LLMTextProcessorVL(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LLMTextProcessorVL",
            display_name="LLM Text Processor VL",
            description="Run a GGUF LLM (optionally with vision/audio/video media) over a prompt.",
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
                io.Boolean.Input("enable_processing", default=True, advanced=True,
                                 tooltip="When enabled, run normal node processing. When disabled, forward the input prompt directly as RESPONSE."),
                io.String.Input("extra_args", multiline=False, default="", advanced=True,
                                tooltip="Optional advanced llama.cpp parameters. Leave empty for normal use."),
                # Reference media slots mirror minimax-h3 counts and order so the LLM sees an identical layout.
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9)),
                io.Image.Input("first_frame", optional=True,
                               tooltip="First frame; becomes the next Picture only when also_ref_first_frame is set."),
                io.Image.Input("last_frame", optional=True,
                               tooltip="Last frame keyframe pin. Never sent as a Picture to the LLM."),
                io.Boolean.Input("also_ref_first_frame", default=False, advanced=True,
                                 tooltip="Also expose first_frame as the next <Picture N> reference."),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video"), prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"), prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3)),
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
    def _ordered_media(
        cls,
        ref_images: dict,
        first_frame,
        also_ref_first_frame: bool,
        ref_videos: dict,
        ref_video_audios: dict,
        ref_audios: dict,
        last_frame=None,
    ):
        # Build three ordered lists matching minimax-h3 reference layout. Order only matters
        # within each type (minimax uses separate <Picture>/<Video>/<Audio> ordinal namespaces).
        def _by_index(mapping):
            items = [(int(key.rsplit("_", 1)[-1]), value) for key, value in mapping.items() if value is not None]
            return [value for _, value in sorted(items, key=lambda pair: pair[0])]

        images_in_order = _by_index(ref_images)
        if also_ref_first_frame and first_frame is not None:
            # Appended after user refs so it becomes the next Picture ordinal, like minimax.
            images_in_order.append(first_frame[:1])

        videos_in_order = _by_index(ref_videos)

        audios_in_order = []
        for key, value in sorted(((k, v) for k, v in ref_videos.items() if v is not None), key=lambda pair: int(pair[0].rsplit("_", 1)[-1])):
            soundtrack_key = "ref_video_audio_" + key.rsplit("_", 1)[-1]
            soundtrack = ref_video_audios.get(soundtrack_key)
            if soundtrack is not None:
                audios_in_order.append(soundtrack)
        audios_in_order.extend(_by_index(ref_audios))

        # last_frame is never a Picture in minimax (keyframe pin only); intentionally unused here.
        return images_in_order, videos_in_order, audios_in_order

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
        ref_images: dict,
        first_frame=None,
        last_frame=None,
        also_ref_first_frame: bool = False,
        ref_videos: dict | None = None,
        ref_video_audios: dict | None = None,
        ref_audios: dict | None = None,
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

        images_in_order, videos_in_order, audios_in_order = cls._ordered_media(
            ref_images=ref_images or {},
            first_frame=first_frame,
            also_ref_first_frame=also_ref_first_frame,
            ref_videos=ref_videos or {},
            ref_video_audios=ref_video_audios or {},
            ref_audios=ref_audios or {},
            last_frame=last_frame,
        )
        command, cleanup_paths = build_command(
            model_path=model_path,
            mmproj_path=mmproj_path,
            system_prompt_path=system_prompt_path,
            images=images_in_order,
            audios=audios_in_order,
            videos=videos_in_order,
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



class LLMTextProcessorVLExtension(ComfyExtension):
    async def get_node_list(self):
        return [LLMTextProcessorVL]
