from collections.abc import Sequence

from sqlalchemy import RowMapping, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.enums import DataBaseType
from backend.core.conf import settings


async def get_tables(db: AsyncSession, table_schema: str) -> Sequence[RowMapping]:
    """
    获取指定 schema 下可被 Text2SQL 挑选的业务表

    复用 code_generator 的反查思路，排除插件自身表（ai_*）与代码生成表（gen_*）。

    :param db: 数据库会话
    :param table_schema: 数据库 schema 名称
    :return:
    """
    if DataBaseType.mysql == settings.DATABASE_TYPE:
        sql = """
        select
          table_name as table_name,
          table_comment as table_comment
        from
          information_schema.tables
        where
          table_schema = :table_schema
          and table_name not like 'ai_%'
          and table_name not like 'gen_%'
        order by
          table_name;
        """
    else:
        sql = """
        select
          c.relname as table_name,
          obj_description (c.oid) as table_comment
        from
          pg_class c
          left join pg_namespace n on n.oid = c.relnamespace
        where
          c.relkind = 'r'
          and n.nspname = :table_schema
          and c.relname not like 'ai_%'
          and c.relname not like 'gen_%'
        order by
          c.relname;
        """
    result = await db.execute(text(sql).bindparams(table_schema=table_schema))
    return result.mappings().all()


async def get_columns(db: AsyncSession, table_schema: str, table_name: str) -> Sequence[RowMapping]:
    """
    获取表的列信息（含主键、可空、注释、类型），用于拼装 DDL 上下文与列预览

    :param db: 数据库会话
    :param table_schema: 数据库 schema 名称
    :param table_name: 表名
    :return:
    """
    if DataBaseType.mysql == settings.DATABASE_TYPE:
        sql = """
        select
          column_name as column_name,
          case
            when column_key = 'pri' then 1
            else 0
          end as is_pk,
          case
            when is_nullable = 'no' or column_key = 'pri' then 0
            else 1
          end as is_nullable,
          ordinal_position as sort,
          column_comment as column_comment,
          column_type as column_type
        from
          information_schema.columns
        where
          table_name = :table_name
          and table_schema = :table_schema
        order by
          sort;
        """
    else:
        sql = """
        select
          a.attname as column_name,
          case
            when exists (
              select 1 from pg_constraint c
              where c.conrelid = t.oid and c.contype = 'p' and a.attnum = any (c.conkey)
            ) then 1
            else 0
          end as is_pk,
          case
            when a.attnotnull or exists (
              select 1 from pg_constraint c
              where c.conrelid = t.oid and c.contype = 'p' and a.attnum = any (c.conkey)
            ) then 0
            else 1
          end as is_nullable,
          a.attnum as sort,
          col_description (t.oid, a.attnum) as column_comment,
          pg_catalog.format_type (a.atttypid, a.atttypmod) as column_type
        from
          pg_attribute a
          join pg_class t on a.attrelid = t.oid
          join pg_namespace n on n.oid = t.relnamespace
        where
          a.attnum > 0
          and not a.attisdropped
          and t.relname = :table_name
          and n.nspname = :table_schema
        order by
          sort;
        """
    result = await db.execute(text(sql).bindparams(table_schema=table_schema, table_name=table_name))
    return result.mappings().all()
