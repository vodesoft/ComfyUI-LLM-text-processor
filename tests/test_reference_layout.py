from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _fake_comfy_modules() -> dict[str, types.ModuleType]:
    model_management = types.ModuleType("comfy.model_management")
    model_management.processing_interrupted = lambda: False
    model_management.throw_exception_if_processing_interrupted = lambda: None
    comfy = types.ModuleType("comfy")
    comfy.model_management = model_management

    # Stub folder_paths so importing nodes (which pulls in folder_registry) succeeds.
    folder_paths_mod = types.ModuleType("folder_paths")
    folder_paths_mod.models_dir = str(PROJECT_ROOT / "models")
    folder_paths_mod.folder_names_and_paths = {}
    folder_paths_mod.get_folder_paths = lambda key: []
    folder_paths_mod.get_filename_list = lambda key: []
    folder_paths_mod.get_full_path = lambda key, name: None

    # Stub the new-API surface so nodes.py imports without real ComfyUI present.
    io_mod = types.ModuleType("comfy_api.io")
    latest = types.ModuleType("comfy_api.latest")
    latest.ComfyExtension = object
    latest.io = io_mod
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest

    return {
        "comfy": comfy,
        "comfy.model_management": model_management,
        "folder_paths": folder_paths_mod,
        "comfy_api": comfy_api,
        "comfy_api.latest": latest,
        "comfy_api.io": io_mod,
    }


def import_nodes():
    package = types.ModuleType("llm_text_processor_testpkg")
    package.__path__ = [str(PROJECT_ROOT)]

    patched = _fake_comfy_modules()
    patched["llm_text_processor_testpkg"] = package
    with mock.patch.dict(sys.modules, patched):
        module = importlib.import_module("llm_text_processor_testpkg.nodes")
    return module


def import_llama_cli():
    package = types.ModuleType("llm_text_processor_testpkg")
    package.__path__ = [str(PROJECT_ROOT)]

    patched = _fake_comfy_modules()
    patched["llm_text_processor_testpkg"] = package
    with mock.patch.dict(sys.modules, patched):
        return importlib.import_module("llm_text_processor_testpkg.llama_cli")


class MediaToken:
    # Minimal stand-in for a ComfyUI IMAGE/Audio tensor; identity is enough for ordering tests.
    def __init__(self, tag: str):
        self.tag = tag

    def __getitem__(self, index):
        return self


class ReferenceLayoutOrderingTests(unittest.TestCase):
    # These pin the minimax-h3 reference layout: separate ordinal namespaces per media type,
    # so order only matters within each list. last_frame is never a Picture.
    def test_images_follow_socket_order(self) -> None:
        nodes = import_nodes()
        ref_images = {
            "ref_image_0": MediaToken("a"),
            "ref_image_1": MediaToken("b"),
            "ref_image_2": None,
            "ref_image_3": MediaToken("c"),
        }
        images, videos, audios = nodes.LLMTextProcessor._ordered_media(
            ref_images=ref_images,
            first_frame=None,
            also_ref_first_frame=False,
            ref_videos={},
            ref_video_audios={},
            ref_audios={},
        )
        self.assertEqual([m.tag for m in images], ["a", "b", "c"])

    def test_first_frame_becomes_next_picture_only_when_flag_set(self) -> None:
        nodes = import_nodes()
        ref_images = {"ref_image_0": MediaToken("a")}
        first = MediaToken("first")

        images_off, _, _ = nodes.LLMTextProcessor._ordered_media(
            ref_images=ref_images,
            first_frame=first,
            also_ref_first_frame=False,
            ref_videos={},
            ref_video_audios={},
            ref_audios={},
        )
        self.assertEqual([m.tag for m in images_off], ["a"])

        images_on, _, _ = nodes.LLMTextProcessor._ordered_media(
            ref_images=ref_images,
            first_frame=first,
            also_ref_first_frame=True,
            ref_videos={},
            ref_video_audios={},
            ref_audios={},
        )
        # Appended after user refs so it is the next Picture ordinal, like minimax.
        self.assertEqual([m.tag for m in images_on], ["a", "first"])

    def test_last_frame_is_never_a_picture(self) -> None:
        nodes = import_nodes()
        images, videos, audios = nodes.LLMTextProcessor._ordered_media(
            ref_images={},
            first_frame=None,
            also_ref_first_frame=False,
            last_frame=MediaToken("last"),
            ref_videos={},
            ref_video_audios={},
            ref_audios={},
        )
        self.assertEqual(images, [])

    def test_audio_soundtrack_precedes_standalone_and_videos_keep_order(self) -> None:
        nodes = import_nodes()
        ref_videos = {
            "ref_video_0": MediaToken("v0"),
            "ref_video_1": MediaToken("v1"),
        }
        ref_video_audios = {"ref_video_audio_1": MediaToken("s1")}
        ref_audios = {"ref_audio_0": MediaToken("standalone0")}

        images, videos, audios = nodes.LLMTextProcessor._ordered_media(
            ref_images={},
            first_frame=None,
            also_ref_first_frame=False,
            last_frame=None,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
        )
        self.assertEqual([m.tag for m in videos], ["v0", "v1"])
        # soundtrack (matched by trailing index) before standalone audios.
        self.assertEqual([m.tag for m in audios], ["s1", "standalone0"])


class CliMediaFlagTests(unittest.TestCase):
    def test_image_audio_video_emitted_as_comma_lists(self) -> None:
        llama_cli = import_llama_cli()

        # Stub encoders so no real files/torch are needed for the token assertions.
        fake_cli = types.SimpleNamespace(cli=Path("llama-cli.exe"))
        with (
            mock.patch.object(llama_cli, "ensure_llama_cli_paths", return_value=fake_cli),
            mock.patch.object(llama_cli, "images_to_temp_pngs", side_effect=lambda imgs: [Path(f"img{i}") for i in range(len(imgs))]),
            mock.patch.object(llama_cli, "audio_to_temp_wav", side_effect=lambda a: Path("aud.wav")),
            mock.patch.object(llama_cli, "video_frames_to_temp_mp4", side_effect=lambda v: Path("vid.mp4")),
        ):

            command, _cleanup = llama_cli.build_command(
                model_path=Path("m.gguf"),
                mmproj_path=Path("p.gguf"),
                system_prompt_path=None,
                images=["img0", "img1"],
                audios=["aud0"],
                videos=["vid0"],
                prompt="hi",
                max_tokens=8,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repeat_penalty=1.0,
                ctx_size=512,
                memory_mode="auto",
                n_gpu_layers=99,
                n_cpu_moe_layers=1,
                seed=1,
                reasoning="off",
            )

        self.assertIn("--image", command)
        self.assertEqual(command[command.index("--image") + 1], "img0,img1")
        self.assertIn("--audio", command)
        self.assertEqual(command[command.index("--audio") + 1], "aud.wav")
        self.assertIn("--video", command)
        self.assertEqual(command[command.index("--video") + 1], "vid.mp4")

    def test_media_without_mmproj_raises(self) -> None:
        llama_cli = import_llama_cli()
        with mock.patch.object(llama_cli, "audio_to_temp_wav", side_effect=lambda a: Path("aud.wav")):
            with self.assertRaises(ValueError):
                llama_cli.build_command(
                    model_path=Path("m.gguf"),
                    mmproj_path=None,
                    system_prompt_path=None,
                    images=[],
                    audios=["aud0"],
                    videos=[],
                    prompt="hi",
                    max_tokens=8,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    repeat_penalty=1.0,
                    ctx_size=512,
                    memory_mode="auto",
                    n_gpu_layers=99,
                    n_cpu_moe_layers=1,
                    seed=1,
                    reasoning="off",
                )


if __name__ == "__main__":
    unittest.main()
