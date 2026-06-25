"""Text2SQL 核心引擎（单次结构化输出，无工具循环）。

流程：预取已选表列信息 + few-shot → 内联进 system_prompt → 复用供应商适配器建
pydantic-ai 模型 → 单次 Agent.run 得到 {sql, summary}（Agent 不执行任何 SQL）→
对最终 SQL 过 sqlglot 护栏并在只读引擎执行取数 → 落历史。

安全边界全部在执行层：Agent 产出的 SQL 不被信任，由 _execute_final 重新过护栏 +
只读账号执行。如此去掉了原先 bespoke 的 list_tables/describe_table/execute_sql
工具循环（与插件既有 capability/tool 体系重复），并收紧攻击面（Agent 再也碰不到执行）。
"""

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.enums import DataBaseType
from backend.common.exception import errors
from backend.common.log import log
from backend.core.conf import settings
from backend.database.db import async_db_session
from backend.plugin.ai.chat.session import AgentSession
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.enums import AIDefaultModelScene
from backend.plugin.ai.model import AIText2SqlHistory, AIText2SqlTable
from backend.plugin.ai.providers.registry import get_provider_adapter
from backend.plugin.ai.service.default_model_service import ai_default_model_service
from backend.plugin.ai.text2sql.guardrails import validate_and_normalize
from backend.plugin.ai.text2sql.readonly_db import get_readonly_session
from backend.plugin.ai.text2sql.schema_meta import get_columns


@dataclass
class Text2SqlDeps:
    """Agent 运行依赖（执行层的只读会话/白名单/超时由 _execute_final 持有，不入 Agent）"""

    schema: str
    tables_info: list[tuple[str, str | None]]  # 启用的已选表 (name, 描述)
    columns_map: dict[str, Any]  # table_name -> 列信息（预取，内联进 system_prompt）
    examples: list[dict[str, str]]


class Text2SqlResult(BaseModel):
    """Agent 结构化输出"""

    sql: str = Field(description='最终只读 SELECT SQL')
    summary: str = Field(description='对查询结果的中文摘要')


def _build_prompt(deps: Text2SqlDeps) -> str:
    blocks = []
    for name, comment in deps.tables_info:
        cols = deps.columns_map.get(name) or []
        col_lines = '\n'.join(
            f"  - {row['column_name']} {row['column_type']}{' [PK]' if row['is_pk'] else ''}"
            f"：{row['column_comment'] or ''}"
            for row in cols
        ) or '  - （无列信息）'
        blocks.append(f'- {name}（{comment or "无注释"}）：\n{col_lines}')
    tables = '\n'.join(blocks) or '- （无）'
    examples = ''
    if deps.examples:
        ex_blocks = '\n'.join(f'问题：{e["question"]}\nSQL：{e["sql"]}' for e in deps.examples)
        examples = f'\n\n参考示例：\n{ex_blocks}'
    return (
        '你是 FBA Text2SQL 助手，将用户的自然语言问题转为只读 SQL，并给出结果摘要。\n'
        f'数据库 schema：{deps.schema}\n'
        f'仅可查询以下表及其列（严禁查询其他表、系统表、信息架构）：\n{tables}{examples}\n\n'
        '规则：\n'
        '1. 只生成单条只读 SELECT（禁止 INSERT/UPDATE/DELETE/DDL/多语句/SELECT INTO）。\n'
        '2. 只查询上面列出的表，按列名与类型构造 SQL。\n'
        '3. 最终输出包含字段：sql（最终 SQL）与 summary（对结果的中文摘要）。'
    )


def _build_agent(model: Any) -> Agent:  # type: ignore[type-arg]
    """单次结构化输出 Agent：无工具，列信息已内联进 system_prompt。

    Agent 不执行任何 SQL——其产出的 sql 字段会被 _execute_final 重新过护栏 + 只读执行，
    因此无需（也不应）在此再提供 execute_sql 工具或自纠正循环。
    """
    agent = Agent(
        model=model,
        deps_type=Text2SqlDeps,
        output_type=Text2SqlResult,
        retries=int(settings.AI_TEXT2SQL_MAX_RETRIES),
    )

    @agent.system_prompt
    def _system_prompt(ctx: RunContext[Text2SqlDeps]) -> str:
        return _build_prompt(ctx.deps)

    return agent


async def _resolve_model(
    *,
    db: AsyncSession,
    provider_id: int | None = None,
    model_id: str | None = None,
) -> tuple[Any, Any]:
    """
    解析 text2sql 使用的供应商与模型

    优先使用调用方传入的 provider_id / model_id（例如聊天会话当前选择的模型）；
    均未传入时回退到默认模型配置（AIDefaultModelScene.text2sql 场景）。

    :param db: 数据库会话
    :param provider_id: 供应商 ID，缺省回退默认模型配置
    :param model_id: 模型 ID，缺省回退默认模型配置
    :return: (provider, model)
    """
    if not provider_id or not model_id:
        default_model = await ai_default_model_service.get(db=db, scene=AIDefaultModelScene.text2sql)
        provider_id = default_model.provider_id
        model_id = default_model.model_id
    provider = await ai_provider_dao.get(db, provider_id)
    if not provider or not provider.status:
        raise errors.RequestError(msg='Text2SQL 供应商不可用，请检查供应商状态')
    model = await ai_model_dao.get_by_model_and_provider(db, model_id, provider_id)
    if not model or not model.status:
        raise errors.RequestError(msg='Text2SQL 模型不可用，请检查模型状态')
    return provider, model


async def _execute_final(
    *,
    readonly_session: AsyncSession,
    sql: str,
    allowlist: set[str],
    max_rows: int,
    default_schema: str,
    dialect: str | None,
    timeout: int,
) -> tuple[list[str], list[dict[str, Any]], int]:
    """
    对最终 SQL 再次过护栏并在只读引擎执行，返回列、数据、总行数

    :raises UnsafeSqlError: 未通过护栏
    :raises TableNotAllowedError: 越表
    :raises RequestError: 执行超时（LIMIT 已由护栏钳制，结果集有界）
    """
    safe = validate_and_normalize(
        sql,
        allowlist=allowlist,
        max_rows=max_rows,
        default_schema=default_schema,
        dialect=dialect,
    )
    try:
        result = await asyncio.wait_for(
            readonly_session.execute(sa_text(safe)),
            timeout=timeout,
        )
    except TimeoutError as exc:  # noqa: PERF203
        raise errors.RequestError(msg=f'Text2SQL 查询执行超时（{timeout}s），请简化查询') from exc
    rows = result.mappings().all()
    columns = list(rows[0].keys()) if rows else []
    data = [dict(row) for row in rows[:max_rows]]
    return columns, data, len(rows)


async def run_query(
    *,
    db: AsyncSession,
    question: str,
    user_id: int,
    selected_tables: Sequence[AIText2SqlTable],
    examples: list[dict[str, str]],
    provider_id: int | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    执行一次 Text2SQL 查询（端到端）

    :param db: 主库会话（用于解析模型，只读）
    :param question: 自然语言问题
    :param user_id: 用户 ID
    :param selected_tables: 启用的已选表
    :param examples: 召回的 few-shot 样例
    :param provider_id: 指定供应商 ID（缺省回退默认模型配置）；聊天 capability 传入会话当前模型
    :param model_id: 指定模型 ID（缺省回退默认模型配置）；聊天 capability 传入会话当前模型
    :return: {sql, summary, columns, rows, row_count, duration_ms, history_id}
    """
    if not settings.AI_TEXT2SQL_ENABLED:
        raise errors.ForbiddenError(msg='Text2SQL 未启用（AI_TEXT2SQL_ENABLED=false）')
    if not selected_tables:
        raise errors.RequestError(msg='尚未挑选任何数据表，请先在数据源管理中挑选')

    schema = settings.AI_TEXT2SQL_SCHEMA
    allowlist = {
        f'{table.schema_name.strip().lower()}.{table.table_name.strip().lower()}'
        for table in selected_tables
    }
    tables_info = [(table.table_name, table.custom_desc or table.table_comment) for table in selected_tables]
    max_rows = int(settings.AI_TEXT2SQL_MAX_ROWS)
    timeout = int(settings.AI_TEXT2SQL_TIMEOUT)
    dialect = 'mysql' if settings.DATABASE_TYPE == DataBaseType.mysql else 'postgres'

    provider, model = await _resolve_model(db=db, provider_id=provider_id, model_id=model_id)
    adapter = get_provider_adapter(provider.type)
    adapter.validate_model_id(model.model_id)

    started = time.perf_counter()
    session = await AgentSession.open(
        adapter=adapter,
        model_name=model.model_id,
        api_key=provider.api_key,
        base_url=provider.api_host,
    )
    final_sql: str | None = None
    summary = ''
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    row_count = 0
    error_msg: str | None = None
    try:
        agent = _build_agent(session.model)
        async with get_readonly_session() as readonly_session:
            # 预取列信息内联进 prompt（替代原 describe_table 工具）；Agent 全程不执行任何 SQL
            columns_map = {
                table.table_name: await get_columns(readonly_session, schema, table.table_name)
                for table in selected_tables
            }
            deps = Text2SqlDeps(
                schema=schema,
                tables_info=tables_info,
                columns_map=columns_map,
                examples=examples,
            )
            run_result = await agent.run(question, deps=deps)
            final_sql = run_result.output.sql
            summary = run_result.output.summary
            columns, rows, row_count = await _execute_final(
                readonly_session=readonly_session,
                sql=final_sql,
                allowlist=allowlist,
                max_rows=max_rows,
                default_schema=schema,
                dialect=dialect,
                timeout=timeout,
            )
    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        log.error(f'Text2SQL 查询失败: {exc}')
        raise
    finally:
        duration_ms = int((time.perf_counter() - started) * 1000)
        await session.aclose()

    # 历史记录（best-effort，不影响主流程）
    history_id = await _write_history(
        user_id=user_id,
        question=question,
        sql=final_sql,
        row_count=row_count,
        duration_ms=duration_ms,
        error=error_msg,
    )

    return {
        'sql': final_sql or '',
        'summary': summary,
        'columns': columns,
        'rows': rows,
        'row_count': row_count,
        'duration_ms': duration_ms,
        'history_id': history_id,
    }


async def _write_history(
    *,
    user_id: int,
    question: str,
    sql: str | None,
    row_count: int,
    duration_ms: int,
    error: str | None,
) -> int | None:
    """best-effort 写入查询历史，失败仅告警"""
    try:
        async with async_db_session.begin() as hist_db:
            record = AIText2SqlHistory(
                user_id=user_id,
                question=question,
                sql=sql,
                executed=0 if error else 1,
                row_count=row_count,
                error=error,
                duration_ms=duration_ms,
            )
            hist_db.add(record)
            await hist_db.flush()
            return record.id
    except Exception as exc:  # noqa: BLE001
        log.warning(f'Text2SQL 历史写入失败: {exc}')
        return None
