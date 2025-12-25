from typing import Literal, TypedDict

from pydantic_ai import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from backend.common.exception import errors


class ChatMessage(TypedDict):
    """发送给浏览器的消息格式"""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def to_chat_message(message: ModelMessage) -> ChatMessage:
    first_part = message.parts[0]
    if isinstance(message, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    elif isinstance(message, ModelResponse) and isinstance(first_part, TextPart):
        return {
            'role': 'model',
            'timestamp': message.timestamp.isoformat(),
            'content': first_part.content,
        }
    raise errors.NotFoundError(msg=f'消息类型错误: {message}')
