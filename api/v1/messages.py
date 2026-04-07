from typing import Annotated

from fastapi import APIRouter, Path, Request
from starlette.responses import StreamingResponse

from backend.common.response.response_schema import ResponseModel, response_base
from backend.common.security.jwt import DependsJwtAuth
from backend.database.db import CurrentSessionTransaction
from backend.plugin.ai.schema.chat import AIChatRegenerateParam
from backend.plugin.ai.schema.message import UpdateAIMessageParam
from backend.plugin.ai.service.message_service import ai_message_service

router = APIRouter()


@router.post(
    '/{pk}/messages/{message_id}/regenerate',
    summary='根据用户消息重生成 AI 回复',
    dependencies=[DependsJwtAuth],
)
async def regenerate_conversation_message(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
    message_id: Annotated[int, Path(gt=0, description='消息 ID')],
    obj: AIChatRegenerateParam,
) -> StreamingResponse:
    return await ai_message_service.regenerate_from_user_message(
        db=db,
        user_id=request.user.id,
        conversation_id=pk,
        message_id=message_id,
        obj=obj,
        accept=request.headers.get('accept'),
    )


@router.post(
    '/{pk}/responses/{message_id}/regenerate',
    summary='根据 AI 回复重生成',
    dependencies=[DependsJwtAuth],
)
async def regenerate_conversation_response(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
    message_id: Annotated[int, Path(gt=0, description='消息 ID')],
    obj: AIChatRegenerateParam,
) -> StreamingResponse:
    return await ai_message_service.regenerate_from_response_message(
        db=db,
        user_id=request.user.id,
        conversation_id=pk,
        message_id=message_id,
        obj=obj,
        accept=request.headers.get('accept'),
    )


@router.put(
    '/{pk}/messages/{message_id}',
    summary='编辑保存指定消息',
    dependencies=[DependsJwtAuth],
)
async def update_conversation_message(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
    message_id: Annotated[int, Path(gt=0, description='消息 ID')],
    obj: UpdateAIMessageParam,
) -> ResponseModel:
    count = await ai_message_service.update(
        db=db,
        conversation_id=pk,
        user_id=request.user.id,
        message_id=message_id,
        obj=obj,
    )
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete(
    '/{pk}/messages',
    summary='清空对话消息',
    dependencies=[DependsJwtAuth],
)
async def clear_conversation_messages(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
) -> ResponseModel:
    count = await ai_message_service.clear(
        db=db,
        conversation_id=pk,
        user_id=request.user.id,
    )
    if count > 0:
        return response_base.success()
    return response_base.fail()


@router.delete(
    '/{pk}/messages/{message_id}',
    summary='删除指定消息',
    dependencies=[DependsJwtAuth],
)
async def delete_conversation_message(
    request: Request,
    db: CurrentSessionTransaction,
    pk: Annotated[str, Path(description='对话 ID')],
    message_id: Annotated[int, Path(gt=0, description='消息 ID')],
) -> ResponseModel:
    count = await ai_message_service.delete(
        db=db,
        conversation_id=pk,
        user_id=request.user.id,
        message_id=message_id,
    )
    if count > 0:
        return response_base.success()
    return response_base.fail()
