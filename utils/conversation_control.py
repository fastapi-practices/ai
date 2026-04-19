def normalize_conversation_title(*, title: str, fallback: str = '新对话') -> str:
    """
    标准化对话标题

    :param title: 原始标题
    :param fallback: 兜底标题
    :return:
    """
    normalized_title = ' '.join(title.split())
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
