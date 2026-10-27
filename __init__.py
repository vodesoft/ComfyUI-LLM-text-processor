from .nodes import LLMTextProcessorVLExtension
from .legacy_processor import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = [
	"LLMTextProcessorVLExtension",
	"NODE_CLASS_MAPPINGS",
	"NODE_DISPLAY_NAME_MAPPINGS",
]


async def comfy_entrypoint():
	return LLMTextProcessorVLExtension()
