import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.app.dependencies.auth import get_current_user
from api.app.dependencies.services import get_chat_service
from api.app.repositories.interfaces import AuthenticatedUser
from api.app.services.chat import ChatService

router = APIRouter(prefix="", tags=["chat"])

class CreateSessionRequest(BaseModel):
    title: str

class ChatMessageRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    content: str
    model_choice: str = "llama3.2:1b"

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_chat_session(
    request: CreateSessionRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
):
    return await service.create_session(user, title=request.title)

@router.get("")
async def list_chat_sessions(
    limit: int = 50,
    user: AuthenticatedUser = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
):
    return await service.list_sessions(user, limit=limit)

@router.get("/{session_id}")
async def get_chat_session(
    session_id: uuid.UUID,
    user: AuthenticatedUser = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
):
    session = await service.get_session(user, session_id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    session_id: uuid.UUID,
    user: AuthenticatedUser = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
):
    deleted = await service.delete_session(user, session_id=session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return None

@router.post("/{session_id}/messages")
async def create_chat_message(
    session_id: uuid.UUID,
    request: ChatMessageRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
):
    async def sse_stream() -> AsyncGenerator[str, None]:
        try:
            async for token, meta, _ in service.stream_message(
                current_user=user,
                session_id=session_id,
                content=request.content,
                model_choice=request.model_choice
            ):
                if token:
                    yield f"event: token\ndata: {json.dumps(token)}\n\n"
                
                if meta:
                    yield f"event: end\ndata: {json.dumps(meta)}\n\n"
                    
        except ValueError as ve:
            yield f"event: error\ndata: {json.dumps(str(ve))}\n\n"
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception("Streaming error")
            yield f"event: error\ndata: {json.dumps(f'Internal server error: {str(e)}')}\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")
