from datetime import datetime

from backend.plugin.ai.model.conversation import AIConversation
from backend.plugin.ai.schema.conversation import UpdateAIConversationParam


def normalize_conversation_title(*, title: str, fallback: str = '新对话') -> str:
    """
    标准化对话标题

    :param title: 原始标题
    :param fallback: 兜底标题
    :return:
    """
    normalized_title = ' '.join(title.split()).strip()
    return normalized_title or fallback


def normalize_generated_conversation_title(*, title: str, fallback: str = '新对话') -> str:
    """
    标准化自动生成的对话标题

    :param title: 原始标题
    :param fallback: 兜底标题
    :return:
    """
    normalized_title = normalize_conversation_title(title=title, fallback=fallback)
    return normalized_title[:253] + '...' if len(normalized_title) > 256 else normalized_title


def build_update_ai_conversation_param(
    *,
    conversation: AIConversation,
    title: str | None = None,
    provider_id: int | None = None,
    model_id: str | None = None,
    context_start_message_id: int | None = None,
    context_cleared_time: datetime | None = None,
) -> UpdateAIConversationParam:
    """
    构建更新对话参数

    :param conversation: 对话对象
    :param title: 对话标题
    :param provider_id: 供应商 ID
    :param model_id: 模型 ID
    :param context_start_message_id: 上下文起始消息 ID
    :param context_cleared_time: 上下文清除时间
    :return:
    """
    return UpdateAIConversationParam(
        conversation_id=conversation.conversation_id,
        title=title or conversation.title,
        provider_id=provider_id or conversation.provider_id,
        model_id=model_id or conversation.model_id,
        user_id=conversation.user_id,
        pinned_time=conversation.pinned_time,
        context_start_message_id=context_start_message_id or conversation.context_start_message_id,
        context_cleared_time=context_cleared_time or conversation.context_cleared_time,
    )
