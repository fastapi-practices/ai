from collections.abc import Sequence

from pydantic_ai import ModelMessage
from pydantic_ai.ui.ag_ui import AGUIAdapter

from backend.plugin.ai.protocol.ag_ui.schema import AIChatAgUiInputMessageParam


def decode_input_messages(*, messages: Sequence[AIChatAgUiInputMessageParam]) -> list[ModelMessage]:
    """
    解析当前轮输入消息列表

    :param messages: 输入消息列表
    :return:
    """
    return AGUIAdapter.load_messages(list(messages), preserve_file_data=True)
