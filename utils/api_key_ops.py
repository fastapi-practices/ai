from typing import Any

_MASK = '********'


def mask_api_key(key: str) -> str:
    """
    隐藏 API Key 中间部分

    :param key: 完整的 API Key
    :return:
    """
    if not key:
        return key
    if len(key) <= 12:
        return _MASK
    return key[:8] + _MASK + key[-4:]


def mask_sensitive_data(value: Any) -> Any:
    """
    递归隐藏敏感配置值

    :param value: 敏感配置
    :return:
    """
    if isinstance(value, str):
        return mask_api_key(value)
    if isinstance(value, dict):
        return {key: mask_sensitive_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_sensitive_data(item) for item in value]
    return value


def restore_masked_sensitive_data(value: Any, current: Any) -> Any:
    """
    将未修改的脱敏值还原为当前敏感配置

    :param value: 更新请求中的配置值
    :param current: 当前持久化配置值
    :return:
    """
    if isinstance(value, str) and isinstance(current, str):
        return current if value == mask_api_key(current) else value
    if isinstance(value, dict) and isinstance(current, dict):
        return {key: restore_masked_sensitive_data(item, current.get(key)) for key, item in value.items()}
    if isinstance(value, list) and isinstance(current, list):
        return [
            restore_masked_sensitive_data(item, current[index] if index < len(current) else None)
            for index, item in enumerate(value)
        ]
    return value
