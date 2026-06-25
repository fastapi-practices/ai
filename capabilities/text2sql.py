import json

from pydantic_ai.capabilities import AbstractCapability, Toolset
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from backend.core.conf import settings
from backend.database.db import async_db_session
from backend.plugin.ai.capabilities.base import function_tools_allowed
from backend.plugin.ai.dataclasses import CapabilityContext, CapabilityResult, ChatAgentDeps
from backend.plugin.ai.enums import AIChatGenerationType
from backend.plugin.ai.service.text2sql_service import text2sql_service


async def build_text2sql_capability(ctx: CapabilityContext) -> CapabilityResult:  # noqa: RUF029
    """
    构建 Text2SQL 能力：让聊天 Agent 按所选数据集自助取数

    当 text2sql_dataset_id 指定了一个数据集、生成类型为文本、且当前组合允许函数工具时，
    注入 text2sql_query 工具，聊天即可回答统计/明细类数据问题。
    复用会话当前选择的供应商与模型，工具仅可见该数据集下启用的表与样例。

    :param ctx: 能力构建上下文
    :return:
    """
    if not settings.AI_TEXT2SQL_ENABLED:
        # 总开关关闭时不向 chat 注入工具（/queries 也在 run_query 处拒绝）
        return CapabilityResult(capability=None)
    dataset_id = ctx.forwarded_props.text2sql_dataset_id
    if not dataset_id:
        return CapabilityResult(capability=None)
    if ctx.forwarded_props.generation_type != AIChatGenerationType.text:
        return CapabilityResult(capability=None)
    if not function_tools_allowed(
        adapter=ctx.adapter,
        supports_tools=ctx.supports_tools,
        has_builtin_tools=ctx.has_builtin_tools,
    ):
        return CapabilityResult(capability=None)
    return CapabilityResult(
        capability=_build_text2sql_toolset(
            provider_id=ctx.forwarded_props.provider_id,
            model_id=ctx.forwarded_props.model_id,
            dataset_id=dataset_id,
        ),
        introduces_function_tool_source=True,
    )


def _build_text2sql_toolset(*, provider_id: int, model_id: str, dataset_id: int) -> AbstractCapability[ChatAgentDeps]:
    """
    装配 text2sql_query 函数工具（复用 run_query，全程只读、过护栏）

    闭包捕获会话当前选择的 provider_id / model_id 与所选数据集 dataset_id；
    取数时仅按 dataset_id 圈定可见表与召回样例。
    run_query 采用函数内懒加载，避免与 chat.session/pipeline 形成模块加载期循环导入。

    :param provider_id: 会话当前选择的供应商 ID
    :param model_id: 会话当前选择的模型 ID
    :param dataset_id: 所选数据集 ID
    :return:
    """
    toolset = FunctionToolset[ChatAgentDeps]()

    @toolset.tool
    async def text2sql_query(ctx: RunContext[ChatAgentDeps], question: str) -> str:
        """
        查询系统数据库中已授权的业务/运营数据（Text2SQL：自然语言转 SQL 只读查询）。

        只要用户的问题涉及"数据库里的数据"——日志、记录、订单、用户、统计、计数、
        排名、趋势、明细、聚合等——都应优先调用本工具，而不是让用户去接外部系统。
        仅查询当前所选数据集中已启用的表，全程只读、受安全护栏保护。

        典型问题（均应调用本工具）：
        - 「今天有多少条操作日志 / 登录日志」「最近的日志记录」
        - 「最近 7 天订单总金额是多少」「每个供应商有多少条记录」
        - 「按金额倒序的前 10 笔订单」「今天/本周/某段时间内 XXX 的数量/计数/汇总」

        :param question: 用户的自然语言数据问题
        :return:
        """
        # 懒加载：engine.run_query → chat.session → chat.pipeline，避免模块加载期循环导入
        from backend.plugin.ai.text2sql.engine import run_query

        async with async_db_session() as db:
            tables = await text2sql_service.get_enabled(db=db, dataset_id=dataset_id)
            if not tables:
                return '当前数据集尚未挑选任何数据表，请先在「数据集」管理中挑选可查询的表。'
            examples = await text2sql_service.get_examples_for(
                db=db,
                tables={table.table_name for table in tables},
                dataset_id=dataset_id,
            )
            result = await run_query(
                db=db,
                question=question,
                user_id=ctx.deps.user_id,
                selected_tables=tables,
                examples=examples,
                provider_id=provider_id,
                model_id=model_id,
            )

        max_rows = int(settings.AI_TEXT2SQL_MAX_ROWS)
        preview = (result.get('rows') or [])[:max_rows]
        return (
            f"SQL：\n{result.get('sql') or ''}\n\n"
            f"摘要：{result.get('summary') or ''}\n"
            f"命中行数：{result.get('row_count', 0)}\n"
            f"列：{result.get('columns') or []}\n"
            f"数据预览：\n{json.dumps(preview, ensure_ascii=False, default=str)}"
        )

    return Toolset(toolset)
