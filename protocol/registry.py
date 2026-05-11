from backend.common.exception import errors
from backend.plugin.ai.protocol.ag_ui.adapter import ag_ui_chat_protocol_adapter
from backend.plugin.ai.protocol.base import ChatProtocolAdapter

_DEFAULT_CHAT_PROTOCOL = 'ag_ui'
_CHAT_PROTOCOL_ADAPTERS: dict[str, ChatProtocolAdapter] = {
    _DEFAULT_CHAT_PROTOCOL: ag_ui_chat_protocol_adapter,
}


def get_chat_protocol_adapter(protocol: str | None = None) -> ChatProtocolAdapter:
    """
    获取聊天协议适配器

    :param protocol: 协议名称
    :return:
    """
    protocol_name = protocol or _DEFAULT_CHAT_PROTOCOL
    adapter = _CHAT_PROTOCOL_ADAPTERS.get(protocol_name)
    if adapter is None:
        raise errors.RequestError(msg='暂不支持的聊天协议')
    return adapter
