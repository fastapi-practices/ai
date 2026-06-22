"""Text2SQL 核心引擎（原生 Pydantic AI）。

流程：解析已挑选表 + few-shot → 复用供应商适配器建 pydantic-ai 模型 → 构建 Agent
（system_prompt + list_tables/describe_table/execute_sql 工具，execute_sql 强制过护栏）
→ 运行得到 {sql, summary} → 对最终 SQL 再次过护栏并在只读引擎执行取数 → 落历史。

注意：pydantic-ai 为 2.x beta，本模块用稳定的装饰器 API（@agent.system_prompt / @agent.tool），
规避构造参数命名差异；上线前务必用真实模型跑通一次（见 README/计划）。
"""

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.exception import errors
from backend.common.log import log
from backend.core.conf import settings
from backend.database.db import async_db_session
from backend.plugin.ai.chat.session import AgentSession
from backend.plugin.ai.crud.crud_model import ai_model_dao
from backend.plugin.ai.crud.crud_provider import ai_provider_dao
from backend.plugin.ai.model import AIText2SqlHistory, AIText2SqlTable
from backend.plugin.ai.providers.registry import get_provider_adapter
from backend.plugin.ai.text2sql.exceptions import TableNotAllowedError, UnsafeSqlError
from backend.plugin.ai.text2sql.guardrails import validate_and_normalize
from backend.plugin.ai.text2sql.readonly_db import get_readonly_session
from backend.plugin.ai.text2sql.schema_meta import get_columns


@dataclass
class Text2SqlDeps:
    """Agent 运行依赖"""

    readonly_session: AsyncSession
    schema: str
    tables: list[tuple[str, str | None]]  # 启用的已选表 (name, 描述)
    allowlist: set[str]
    examples: list[dict[str, str]]
    max_rows: int
    timeout: int


class Text2SqlResult(BaseModel):
    """Agent 结构化输出"""

    sql: str = Field(description='最终只读 SELECT SQL')
    summary: str = Field(description='对查询结果的中文摘要')


def _build_prompt(deps: Text2SqlDeps) -> str:
    tables = '\n'.join(f'- {name}：{comment or "（无注释）"}' for name, comment in deps.tables) or '- （无）'
    examples = ''
    if deps.examples:
        blocks = '\n'.join(f'问题：{e["question"]}\nSQL：{e["sql"]}' for e in deps.examples)
        examples = f'\n\n参考示例：\n{blocks}'
    return (
        '你是 FBA Text2SQL 助手，将用户的自然语言问题转为只读 SQL，并给出结果摘要。\n'
        f'数据库 schema：{deps.schema}\n'
        f'仅可查询以下表（严禁查询其他表）：\n{tables}{examples}\n\n'
        '规则：\n'
        '1. 只生成只读 SELECT（禁止 INSERT/UPDATE/DELETE/DDL/多语句/SELECT INTO）。\n'
        '2. 只查询上面列出的表。\n'
        '3. 需要列信息时调用 describe_table；验证查询时调用 execute_sql。\n'
        '4. 若 execute_sql 报错，请根据错误修正 SQL 后重试。\n'
        '5. 最终输出包含字段：sql（最终 SQL）与 summary（对结果的中文摘要）。'
    )


def _build_agent(model: Any) -> Agent:  # type: ignore[type-arg]
    agent = Agent(
        model=model,
        deps_type=Text2SqlDeps,
        output_type=Text2SqlResult,
        retries=int(settings.AI_TEXT2SQL_MAX_RETRIES),
    )

    @agent.system_prompt
    def _system_prompt(ctx: RunContext[Text2SqlDeps]) -> str:
        return _build_prompt(ctx.deps)

    @agent.tool
    async def list_tables(ctx: RunContext[Text2SqlDeps]) -> str:
        items = '\n'.join(f'- {name}：{comment or "（无注释）"}' for name, comment in ctx.deps.tables)
        return f'可查询表：\n{items}'

    @agent.tool
    async def describe_table(ctx: RunContext[Text2SqlDeps], table_name: str) -> str:
        if table_name.lower() not in ctx.deps.allowlist:
            return f'表 {table_name} 不在可查询范围内'
        rows = await get_columns(ctx.deps.readonly_session, ctx.deps.schema, table_name)
        lines = [
            f"- {row['column_name']} {row['column_type']}{' [PK]' if row['is_pk'] else ''}：{row['column_comment'] or ''}"
            for row in rows
        ]
        return f'表 {table_name} 列：\n' + '\n'.join(lines)

    @agent.tool
    async def execute_sql(ctx: RunContext[Text2SqlDeps], sql: str) -> str:
        try:
            safe = validate_and_normalize(sql, allowlist=ctx.deps.allowlist, max_rows=ctx.deps.max_rows)
        except (UnsafeSqlError, TableNotAllowedError) as exc:
            return f'校验失败，请改写：{exc}'
        try:
            result = await asyncio.wait_for(
                ctx.deps.readonly_session.execute(sa_text(safe)),
                timeout=ctx.deps.timeout,
            )
        except TimeoutError:
            return '执行超时，请简化查询'
        except Exception as exc:  # noqa: BLE001
            return f'执行失败，请检查 SQL：{exc}'
        rows = result.mappings().all()
        columns = list(rows[0].keys()) if rows else []
        preview = [dict(row) for row in rows[:20]]
        return (
            f'列：{columns}\n命中行数：{len(rows)}\n'
            f'预览（前 {len(preview)} 行）：\n{json.dumps(preview, ensure_ascii=False, default=str)}'
        )

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
    均未传入时回退到 .env 配置的 AI_TEXT2SQL_PROVIDER_ID / AI_TEXT2SQL_MODEL_ID。

    :param db: 数据库会话
    :param provider_id: 供应商 ID，缺省回退配置
    :param model_id: 模型 ID，缺省回退配置
    :return: (provider, model)
    """
    if not provider_id or not model_id:
        provider_id = settings.AI_TEXT2SQL_PROVIDER_ID
        model_id = settings.AI_TEXT2SQL_MODEL_ID
    if not provider_id or not model_id:
        raise errors.RequestError(msg='未配置 Text2SQL 模型，请在 .env 设置 AI_TEXT2SQL_PROVIDER_ID 与 AI_TEXT2SQL_MODEL_ID')
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
) -> tuple[list[str], list[dict[str, Any]], int]:
    """
    对最终 SQL 再次过护栏并在只读引擎执行，返回列、数据、总行数

    :raises UnsafeSqlError: 未通过护栏
    :raises TableNotAllowedError: 越表
    """
    safe = validate_and_normalize(sql, allowlist=allowlist, max_rows=max_rows)
    result = await readonly_session.execute(sa_text(safe))
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
    :param provider_id: 指定供应商 ID（缺省回退配置）；聊天 capability 传入会话当前模型
    :param model_id: 指定模型 ID（缺省回退配置）；聊天 capability 传入会话当前模型
    :return: {sql, summary, columns, rows, row_count, duration_ms, history_id}
    """
    if not selected_tables:
        raise errors.RequestError(msg='尚未挑选任何数据表，请先在数据源管理中挑选')

    schema = settings.AI_TEXT2SQL_SCHEMA
    allowlist = {table.table_name.lower() for table in selected_tables}
    tables_info = [(table.table_name, table.custom_desc or table.table_comment) for table in selected_tables]
    max_rows = int(settings.AI_TEXT2SQL_MAX_ROWS)
    timeout = int(settings.AI_TEXT2SQL_TIMEOUT)

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
            deps = Text2SqlDeps(
                readonly_session=readonly_session,
                schema=schema,
                tables=tables_info,
                allowlist=allowlist,
                examples=examples,
                max_rows=max_rows,
                timeout=timeout,
            )
            run_result = await agent.run(question, deps=deps)
            final_sql = run_result.output.sql
            summary = run_result.output.summary
            columns, rows, row_count = await _execute_final(
                readonly_session=readonly_session,
                sql=final_sql,
                allowlist=allowlist,
                max_rows=max_rows,
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
