from typing import Any

from pydantic_ai.builtin_tools import ImageGenerationTool
from pydantic_ai.capabilities import AbstractCapability, BuiltinTool

from backend.plugin.ai.enums import AIProviderType
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam

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
GOOGLE_IMAGE_GENERATION_FIELDS = {
    'image_output_compression',
    'image_output_format',
    'image_size',
    'image_aspect_ratio',
}


def build_image_generation_capability(
    *,
    forwarded_props: AIChatForwardedPropsParam,
    provider_type: int,
) -> AbstractCapability[Any]:
    """
    构建图片生成能力

    :param forwarded_props: 聊天扩展参数
    :param provider_type: 供应商类型
    :return:
    """
    image_fields = (
        GOOGLE_IMAGE_GENERATION_FIELDS
        if AIProviderType(provider_type) == AIProviderType.google
        else set(IMAGE_GENERATION_FIELD_MAP)
    )
    image_settings = forwarded_props.model_dump(include=image_fields, exclude_unset=True, exclude_none=True)
    image_tool_settings = {IMAGE_GENERATION_FIELD_MAP[field]: value for field, value in image_settings.items()}
    return BuiltinTool(ImageGenerationTool(**image_tool_settings))
