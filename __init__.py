from .nodes import LLMTextProcessorVLExtension
__all__ = [
	"LLMTextProcessorVLExtension",
]


async def comfy_entrypoint():
	return LLMTextProcessorVLExtension()
