from .nodes import LLMTextProcessorExtension

__all__ = ["LLMTextProcessorExtension"]


async def comfy_entrypoint():
	return LLMTextProcessorExtension()
