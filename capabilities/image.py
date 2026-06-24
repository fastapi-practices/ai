from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import ImageGenerationTool

from backend.common.exception import errors
from backend.plugin.ai.dataclasses import CapabilityContext, CapabilityResult
from backend.plugin.ai.enums import AIChatGenerationType

IMAGE_GENERATION_FIELD_MAP: dict[str, str] = {
    'image_action': 'action',
    'image_background': 'background',
    'image_input_fidelity': 'input_fidelity',
    'image_moderation': 'moderation',
    'image_model': 'model',
    'image_output_compression': 'output_compression',
    'image_output_format': 'output_format',
    'image_partial_images': 'partial_images',
    'image_quality': 'quality',
    'image_size': 'size',
    'image_aspect_ratio': 'aspect_ratio',
}


async def build_image_generation_capability(ctx: CapabilityContext) -> CapabilityResult:  # noqa: RUF029
    """
    构建图片生成能力

    :param ctx: 能力构建上下文
    :return:
    """
    if ctx.forwarded_props.generation_type != AIChatGenerationType.image:
        return CapabilityResult(capability=None)
    if not ctx.adapter.capabilities['supports_image_generation']:
        raise errors.RequestError(msg='当前模型暂不支持图片生成，请更换模型')
    if not ctx.supports_image_output:
        raise errors.RequestError(msg='当前模型暂不支持图片生成，请更换模型')

    image_supported = ctx.adapter.capabilities['image_supported_fields']
    image_fields = image_supported if image_supported is not None else frozenset(IMAGE_GENERATION_FIELD_MAP)
    image_settings = ctx.forwarded_props.model_dump(include=image_fields, exclude_unset=True, exclude_none=True)
    image_tool_settings = {IMAGE_GENERATION_FIELD_MAP[field]: value for field, value in image_settings.items()}

    return CapabilityResult(
        capability=NativeTool(ImageGenerationTool(**image_tool_settings)),
        introduces_builtin_tool=True,
    )
