import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.app.models.entities import ChatSession, ChatMessage, MessageRole, MessageStatus
from api.app.repositories.interfaces import ChatRepository, ChatSessionRecord, ChatMessageRecord


def _enum_value(value: str | MessageRole | MessageStatus) -> str:
    return getattr(value, "value", value)

@dataclass
class ChatMessageRecord:
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    status: str
    model_name: str | None
    retrieval_used: bool
    metadata_json: dict | None
    created_at: datetime

@dataclass
class ChatSessionRecord:
    id: uuid.UUID
    workspace_id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageRecord] | None = None

class SqlAlchemyChatRepository(ChatRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create_session(self, workspace_id: uuid.UUID, title: str) -> ChatSessionRecord:
        chat_session = ChatSession(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            title=title
        )
        self._session.add(chat_session)
        await self._session.commit()
        await self._session.refresh(chat_session)
        return ChatSessionRecord(
            id=chat_session.id,
            workspace_id=chat_session.workspace_id,
            title=chat_session.title,
            created_at=chat_session.created_at,
            updated_at=chat_session.updated_at
        )

    async def get_session(self, workspace_id: uuid.UUID, session_id: uuid.UUID) -> ChatSessionRecord | None:
        statement = select(ChatSession).where(
            ChatSession.workspace_id == workspace_id,
            ChatSession.id == session_id
        ).options(selectinload(ChatSession.messages))
        chat_session = await self._session.scalar(statement)
        if not chat_session:
            return None
        
        # Sort messages by created_at explicitly just to be safe, handling None values
        sorted_msgs = sorted(chat_session.messages, key=lambda m: m.created_at or datetime.min)
        messages = [
            ChatMessageRecord(
                id=m.id,
                session_id=m.session_id,
                role=_enum_value(m.role),
                content=m.content,
                status=_enum_value(m.status),
                model_name=m.model_name,
                retrieval_used=m.retrieval_used,
                metadata_json=m.metadata_json,
                created_at=m.created_at
            )
            for m in sorted_msgs
        ]
        
        return ChatSessionRecord(
            id=chat_session.id,
            workspace_id=chat_session.workspace_id,
            title=chat_session.title,
            created_at=chat_session.created_at,
            updated_at=chat_session.updated_at,
            messages=messages
        )

    async def list_sessions(self, workspace_id: uuid.UUID, limit: int = 50) -> list[ChatSessionRecord]:
        statement = select(ChatSession).where(
            ChatSession.workspace_id == workspace_id
        ).order_by(ChatSession.created_at.desc()).limit(limit)
        
        sessions = (await self._session.scalars(statement)).all()
        return [
            ChatSessionRecord(
                id=s.id,
                workspace_id=s.workspace_id,
                title=s.title,
                created_at=s.created_at,
                updated_at=s.updated_at
            )
            for s in sessions
        ]

    async def delete_session(self, workspace_id: uuid.UUID, session_id: uuid.UUID) -> bool:
        statement = select(ChatSession).where(
            ChatSession.workspace_id == workspace_id,
            ChatSession.id == session_id,
        )
        chat_session = await self._session.scalar(statement)
        if not chat_session:
            return False

        await self._session.delete(chat_session)
        await self._session.commit()
        return True

    async def create_message(
        self, 
        session_id: uuid.UUID, 
        role: MessageRole, 
        content: str, 
        status: MessageStatus,
        model_name: str | None = None,
        retrieval_used: bool = False,
        metadata_json: dict | None = None,
        commit: bool = True
    ) -> ChatMessageRecord:
        msg = ChatMessage(
            id=uuid.uuid4(),
            session_id=session_id,
            role=role,
            content=content,
            status=status,
            model_name=model_name,
            retrieval_used=retrieval_used,
            metadata_json=metadata_json
        )
        self._session.add(msg)
        
        if commit:
            await self._session.commit()
            await self._session.refresh(msg)
        else:
            await self._session.flush()
            
        return ChatMessageRecord(
            id=msg.id,
            session_id=msg.session_id,
            role=_enum_value(msg.role),
            content=msg.content,
            status=_enum_value(msg.status),
            model_name=msg.model_name,
            retrieval_used=msg.retrieval_used,
            metadata_json=msg.metadata_json,
            created_at=msg.created_at
        )

    async def update_message(
        self,
        message_id: uuid.UUID,
        content: str,
        status: MessageStatus,
        metadata_json: dict | None = None
    ) -> ChatMessageRecord | None:
        statement = select(ChatMessage).where(ChatMessage.id == message_id)
        msg = await self._session.scalar(statement)
        if not msg:
            return None
            
        msg.content = content
        msg.status = status
        if metadata_json is not None:
            msg.metadata_json = metadata_json
            
        await self._session.commit()
        await self._session.refresh(msg)
        
        return ChatMessageRecord(
            id=msg.id,
            session_id=msg.session_id,
            role=_enum_value(msg.role),
            content=msg.content,
            status=_enum_value(msg.status),
            model_name=msg.model_name,
            retrieval_used=msg.retrieval_used,
            metadata_json=msg.metadata_json,
            created_at=msg.created_at
        )
