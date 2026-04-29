from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from api.app.models.entities import ChatMessage, ChatSession, Workspace
from api.app.services.chat import ChatService
from api.app.services.router import RouteResult


@pytest.fixture
def mock_repo():
    repo = AsyncMock()
    repo.ensure_workspace_for_user.return_value = Workspace(id=uuid4(), name="Test WS")
    return repo


@pytest.fixture
def mock_chat_repo():
    repo = AsyncMock()
    session = ChatSession(id=uuid4(), workspace_id=uuid4(), title="Test Session")
    repo.get_session.return_value = session
    repo.create_message.return_value = ChatMessage(
        id=uuid4(),
        session_id=session.id,
        role="assistant",
        content="",
    )
    return repo


@pytest.fixture
def mock_rag_service():
    service = AsyncMock()
    service.retrieve_context.return_value = []

    async def stream_generation(*args, **kwargs):
        yield "RAG response token", None
        yield "", {"provider": "gemini", "route": "rag", "citations": []}

    service.stream_generation = stream_generation
    return service


@pytest.fixture
def mock_sql_service():
    service = AsyncMock()

    async def stream_generation(*args, **kwargs):
        yield "SQL response token", None
        yield "", {"provider": "gemini", "route": "sql", "sql_used": "SELECT *", "row_count": 5}

    service.stream_generation = stream_generation
    return service


@pytest.fixture
def mock_general_chat_service():
    service = AsyncMock()

    async def stream_generation(*args, **kwargs):
        yield "General chat token", None
        yield "", {
            "provider": {"name": "ollama", "model": "llama3.2:1b"},
            "route": "chat",
            "retrieval_used": False,
            "citations": [],
            "error": None,
        }

    service.stream_generation = stream_generation
    return service


@pytest.fixture
def mock_router_service():
    service = AsyncMock()
    service.decide_route.return_value = RouteResult(route="rag", reason="Default mock", confidence=0.99)
    return service


def build_service(
    mock_repo,
    mock_chat_repo,
    mock_rag_service,
    mock_sql_service,
    mock_general_chat_service,
    mock_router_service,
):
    return ChatService(
        repository=mock_repo,
        chat_repository=mock_chat_repo,
        rag_service=mock_rag_service,
        sql_service=mock_sql_service,
        general_chat_service=mock_general_chat_service,
        router_service=mock_router_service,
    )


async def collect_stream(stream):
    tokens = []
    metas = []
    async for token, meta, _ in stream:
        if token:
            tokens.append(token)
        if meta:
            metas.append(meta)
    return tokens, metas


@pytest.mark.anyio
async def test_chat_service_routes_to_rag(
    mock_repo, mock_chat_repo, mock_rag_service, mock_sql_service, mock_general_chat_service, mock_router_service
):
    service = build_service(
        mock_repo,
        mock_chat_repo,
        mock_rag_service,
        mock_sql_service,
        mock_general_chat_service,
        mock_router_service,
    )

    tokens, metas = await collect_stream(
        service.stream_message(MagicMock(), uuid4(), "What's in my document?", "llama3.2:1b")
    )

    assert "RAG response token" in tokens
    assert metas[0]["route"] == "rag"
    mock_rag_service.retrieve_context.assert_called_once()
    mock_chat_repo.update_message.assert_called_once()


@pytest.mark.anyio
async def test_chat_service_routes_to_chat_without_retrieval(
    mock_repo, mock_chat_repo, mock_rag_service, mock_sql_service, mock_general_chat_service, mock_router_service
):
    mock_router_service.decide_route.return_value = RouteResult(route="chat", reason="Greeting", confidence=0.91)
    service = build_service(
        mock_repo,
        mock_chat_repo,
        mock_rag_service,
        mock_sql_service,
        mock_general_chat_service,
        mock_router_service,
    )

    tokens, metas = await collect_stream(
        service.stream_message(MagicMock(), uuid4(), "Hello", "llama3.2:1b")
    )

    assert "General chat token" in tokens
    assert metas[0]["route"] == "chat"
    assert metas[0]["retrieval_used"] is False
    assert metas[0]["citations"] == []
    mock_rag_service.retrieve_context.assert_not_called()
    mock_chat_repo.update_message.assert_called_once()


@pytest.mark.anyio
async def test_chat_service_routes_to_sql(
    mock_repo, mock_chat_repo, mock_rag_service, mock_sql_service, mock_general_chat_service, mock_router_service
):
    mock_router_service.decide_route.return_value = RouteResult(route="sql", reason="Requires table", confidence=0.88)
    service = build_service(
        mock_repo,
        mock_chat_repo,
        mock_rag_service,
        mock_sql_service,
        mock_general_chat_service,
        mock_router_service,
    )

    tokens, metas = await collect_stream(
        service.stream_message(MagicMock(), uuid4(), "Tổng doanh thu", "llama3.2:1b")
    )

    assert "SQL response token" in tokens
    assert metas[0]["route"] == "sql"
    assert metas[0]["sql_used"] == "SELECT *"
    mock_rag_service.retrieve_context.assert_not_called()
    mock_chat_repo.update_message.assert_called_once()


@pytest.mark.anyio
async def test_chat_service_handles_clarification(
    mock_repo, mock_chat_repo, mock_rag_service, mock_sql_service, mock_general_chat_service, mock_router_service
):
    mock_router_service.decide_route.return_value = RouteResult(route="clarification", reason="Ambiguous", confidence=0.9)
    service = build_service(
        mock_repo,
        mock_chat_repo,
        mock_rag_service,
        mock_sql_service,
        mock_general_chat_service,
        mock_router_service,
    )

    tokens, metas = await collect_stream(
        service.stream_message(MagicMock(), uuid4(), "???", "llama3.2:1b")
    )

    assert "chưa đủ rõ ràng" in tokens[0]
    assert metas[0]["route"] == "clarification"
    mock_rag_service.retrieve_context.assert_not_called()
