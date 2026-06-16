from typing import cast

from pydantic_ai import ModelSettings

from backend.plugin.ai.providers.base import COMMON_MODEL_SETTING_FIELDS, ProviderAdapter
from backend.plugin.ai.schema.chat import AIChatForwardedPropsParam


def build_model_settings(
    *,
    adapter: ProviderAdapter,
    forwarded_props: AIChatForwardedPropsParam,
) -> ModelSettings:
    """
    按供应商构建模型配置

    :param adapter: 供应商适配器
    :param forwarded_props: 聊天扩展参数
    :return:
    """
    included = COMMON_MODEL_SETTING_FIELDS - adapter.capabilities['excluded_setting_fields']
    payload = forwarded_props.model_dump(include=included, exclude_unset=True, exclude_none=True)
    payload.update(adapter.resolve_extra_settings(forwarded_props=forwarded_props))
    return cast('ModelSettings', adapter.settings_cls(**payload))
