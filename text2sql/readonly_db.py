from typing import Any

from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from backend.common.enums import DataBaseType
from backend.common.exception import errors
from backend.common.log import log
from backend.core.conf import settings

# 只读引擎与会话工厂（懒加载）
_readonly_engine: AsyncEngine | None = None
_readonly_session_maker: async_sessionmaker[AsyncSession | Any] | None = None


def _readonly_url_or_none() -> URL | None:
    """
    根据只读账号配置生成连接 URL

    :return: 配置完整时返回只读连接 URL，否则返回 None（调用方回退主库）
    """
    host = settings.AI_TEXT2SQL_READONLY_HOST
    user = settings.AI_TEXT2SQL_READONLY_USER
    password = settings.AI_TEXT2SQL_READONLY_PASSWORD
    if not host or not user:
        return None

    driver = 'mysql+asyncmy' if DataBaseType.mysql == settings.DATABASE_TYPE else 'postgresql+asyncpg'
    url = URL.create(
        drivername=driver,
        username=user,
        password=password,
        host=host,
        port=settings.AI_TEXT2SQL_READONLY_PORT or settings.DATABASE_PORT,
        database=settings.DATABASE_SCHEMA,
    )
    if DataBaseType.mysql == settings.DATABASE_TYPE:
        url = url.update_query_dict({'charset': settings.DATABASE_CHARSET})
    return url


def get_readonly_engine() -> AsyncEngine:
    """
    获取只读引擎

    fail-closed：未配置只读账号时**拒绝回退主库**——执行 LLM 生成的 SQL 必须落在
    显式配置、仅 SELECT 权限的只读账号上。主库可写，绝不可作为 Text2SQL 的执行目标。

    :return: 只读异步引擎
    :raises RequestError: 未配置只读账号
    """
    global _readonly_engine, _readonly_session_maker
    if _readonly_engine is not None:
        return _readonly_engine

    url = _readonly_url_or_none()
    if url is None:
        raise errors.RequestError(
            msg='AI Text2SQL 未配置只读账号（AI_TEXT2SQL_READONLY_HOST/USER/PASSWORD），'
            '拒绝执行：执行 LLM 生成的 SQL 必须使用仅 SELECT 权限的只读账号',
        )

    _readonly_engine = create_async_engine(
        url,
        future=True,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,
        pool_pre_ping=True,
    )

    _readonly_session_maker = async_sessionmaker(
        bind=_readonly_engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )
    return _readonly_engine


def get_readonly_session() -> AsyncSession:
    """
    获取只读会话（上下文管理器）

    用法：`async with get_readonly_session() as session: ...`

    :return: 只读异步会话
    """
    if _readonly_session_maker is None:
        get_readonly_engine()
    assert _readonly_session_maker is not None
    return _readonly_session_maker()
