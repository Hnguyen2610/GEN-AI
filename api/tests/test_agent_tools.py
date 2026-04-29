from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from api.app.agent.tools import AgentToolRegistry


@pytest.fixture
def mock_rag():
    rag = AsyncMock()
    context = MagicMock()
    context.original_filename = "doc.txt"
    context.asset_id = uuid4()
    context.chunk_id = uuid4()
    context.content = "This is a document."
    context.source_page = None
    rag.retrieve_context.return_value = [context]
    return rag


@pytest.fixture
def mock_sql():
    sql = AsyncMock()
    sql.list_assets.return_value = [{"title": "Sales", "kind": "dataset"}]
    sql.get_workspace_schema_context.return_value = (
        "Schema content",
        [SimpleNamespace(id=uuid4(), title="sales")],
    )
    sql.get_dataset_profile.return_value = {"schema": "Schema content"}
    sql.preview_rows.return_value = {"rows": [{"a": 1}], "row_count": 1}
    sql.execute_readonly_sql = MagicMock(return_value={"rows": [{"a": 1}], "row_count": 1, "sql_used": "SELECT *"})
    return sql


@pytest.mark.anyio
async def test_search_knowledge_tool(mock_rag, mock_sql):
    registry = AgentToolRegistry(uuid4(), mock_rag, mock_sql)
    result = await registry.search_knowledge("test")

    assert len(result["result"]) == 1
    assert result["result"][0]["document"] == "doc.txt"


@pytest.mark.anyio
async def test_search_knowledge_empty(mock_rag, mock_sql):
    mock_rag.retrieve_context.return_value = []
    registry = AgentToolRegistry(uuid4(), mock_rag, mock_sql)
    result = await registry.search_knowledge("test")

    assert "No relevant documents" in result["result"]


@pytest.mark.anyio
async def test_get_dataset_schema_tool(mock_rag, mock_sql):
    registry = AgentToolRegistry(uuid4(), mock_rag, mock_sql)
    result = await registry.get_dataset_schema()

    assert result["result"] == "Schema content"
    assert len(registry._dataset_assets) == 1


@pytest.mark.anyio
async def test_run_duckdb_sql_tool(mock_rag, mock_sql):
    registry = AgentToolRegistry(uuid4(), mock_rag, mock_sql)
    result = await registry.run_duckdb_sql("SELECT *")

    assert result["total_rows"] == 1
    assert result["data"][0]["a"] == 1
    mock_sql.execute_readonly_sql.assert_called_once()
    assert mock_sql.execute_readonly_sql.call_args.args[2] == registry._dataset_assets


@pytest.mark.anyio
async def test_agent_support_tools(mock_rag, mock_sql):
    registry = AgentToolRegistry(uuid4(), mock_rag, mock_sql)

    assert (await registry.list_assets())["result"][0]["title"] == "Sales"
    assert (await registry.get_dataset_profile())["result"]["schema"] == "Schema content"
    assert (await registry.preview_rows("Sheet1", 3))["row_count"] == 1
    assert "cung cấp" in (await registry.ask_for_clarification(""))["result"]
