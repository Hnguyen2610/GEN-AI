import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from google.genai import types

from api.app.agent.pev import PEVAgentService
from api.app.agent.tools import AgentToolRegistry

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.gemini_api_key = "test_key"
    return settings

@pytest.fixture
def mock_registry():
    registry = AsyncMock(spec=AgentToolRegistry)
    registry.search_knowledge.return_value = {"result": "Mock document output"}
    registry.get_dataset_schema.return_value = {"result": "Mock schema"}
    registry.run_duckdb_sql.return_value = {"total_rows": 1, "data": [{"id": 1}]}
    return registry


@pytest.mark.anyio
@patch('api.app.agent.pev.genai.Client')
async def test_agent_resolves_without_tools(mock_client_cls, mock_settings, mock_registry):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.text = "Hello there!"
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = [MagicMock(function_call=None)]
    
    mock_aio = AsyncMock()
    mock_aio.models.generate_content.return_value = mock_response
    mock_client.aio = mock_aio
    
    service = PEVAgentService(settings=mock_settings)
    
    generator = service.stream_response(
        workspace_id=uuid4(), 
        query="Hi", 
        history=[], 
        model_name="gemini-2.5-flash",
        registry=mock_registry
    )
    
    tokens = []
    metas = []
    async for token, meta in generator:
        if token:
            tokens.append(token)
        if meta:
            metas.append(meta)
            
    assert "Hello there!" in tokens
    assert len(metas) == 1
    assert metas[0]["route"] == "agent"
    assert metas[0]["steps_taken"] == 1


@pytest.mark.anyio
@patch('api.app.agent.pev.genai.Client')
async def test_agent_resolves_with_tools(mock_client_cls, mock_settings, mock_registry):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    # 1. First iteration requests tool
    tool_part = MagicMock()
    tool_part.function_call = MagicMock(name="search_knowledge", args={"query": "test query"})
    tool_part.function_call.name = "search_knowledge"
    
    resp1 = MagicMock()
    resp1.text = ""
    resp1.candidates = [MagicMock()]
    resp1.candidates[0].content = MagicMock()
    resp1.candidates[0].content.parts = [tool_part]
    
    # 2. Second iteration concludes
    resp2 = MagicMock()
    resp2.text = "The answer is XYZ."
    resp2.candidates = [MagicMock()]
    resp2.candidates[0].content = MagicMock()
    resp2.candidates[0].content.parts = [MagicMock(function_call=None)]
    
    mock_aio = AsyncMock()
    mock_aio.models.generate_content.side_effect = [resp1, resp2]
    mock_client.aio = mock_aio
    
    service = PEVAgentService(settings=mock_settings)
    
    generator = service.stream_response(
        workspace_id=uuid4(), 
        query="What is test?", 
        history=[], 
        model_name="gemini-2.5-flash",
        registry=mock_registry
    )
    
    tokens = []
    metas = []
    async for token, meta in generator:
        if token:
            tokens.append(token)
        if meta:
            metas.append(meta)
            
    assert "The answer is XYZ." in tokens
    assert len(metas) == 1
    assert metas[0]["steps_taken"] == 2
    assert len(metas[0]["agent_traces"]) == 1
    assert metas[0]["agent_traces"][0]["tool"] == "search_knowledge"
    mock_registry.search_knowledge.assert_called_once()
