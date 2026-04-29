from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from api.app.models.entities import ChatMessage, ChatSession
from api.app.repositories.chat import SqlAlchemyChatRepository


class _FakeSession:
    def __init__(self, chat_session: ChatSession):
        self._chat_session = chat_session
        self.deleted = None
        self.committed = False

    async def scalar(self, _statement):
        return self._chat_session

    async def delete(self, chat_session):
        self.deleted = chat_session

    async def commit(self):
        self.committed = True


@pytest.mark.asyncio
async def test_get_session_handles_string_role_and_status_from_db():
    workspace_id = uuid4()
    session_id = uuid4()
    now = datetime.now(timezone.utc)

    session = ChatSession(
        id=session_id,
        workspace_id=workspace_id,
        title="debug session",
        created_at=now,
        updated_at=now,
    )
    later = now + timedelta(seconds=1)
    session.messages = [
        ChatMessage(
            id=uuid4(),
            session_id=session_id,
            role="assistant",
            content="done",
            status="completed",
            model_name="llama3.2:1b",
            retrieval_used=True,
            metadata_json=None,
            created_at=later,
        ),
        ChatMessage(
            id=uuid4(),
            session_id=session_id,
            role="user",
            content="hello",
            status="completed",
            model_name=None,
            retrieval_used=False,
            metadata_json=None,
            created_at=now,
        ),
    ]

    repository = SqlAlchemyChatRepository(_FakeSession(session))

    record = await repository.get_session(workspace_id, session_id)

    assert record is not None
    assert [message.role for message in record.messages] == ["user", "assistant"]
    assert [message.status for message in record.messages] == ["completed", "completed"]


@pytest.mark.asyncio
async def test_delete_session_removes_matching_workspace_session():
    workspace_id = uuid4()
    session_id = uuid4()
    now = datetime.now(timezone.utc)

    session = ChatSession(
        id=session_id,
        workspace_id=workspace_id,
        title="delete me",
        created_at=now,
        updated_at=now,
    )

    fake_session = _FakeSession(session)
    repository = SqlAlchemyChatRepository(fake_session)

    deleted = await repository.delete_session(workspace_id, session_id)

    assert deleted is True
    assert fake_session.deleted is session
    assert fake_session.committed is True
