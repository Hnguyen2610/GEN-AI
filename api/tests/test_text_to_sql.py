import json
from types import SimpleNamespace
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from api.app.services.sql import SQLSafetyError, TextToSQLService

def test_sql_safety_check_valid():
    service = TextToSQLService(settings=MagicMock(), session=MagicMock())
    assert service._is_safe_sql("SELECT * FROM table") is True
    assert service._is_safe_sql("WITH cte AS (SELECT 1) SELECT * FROM cte") is True
    assert service._is_safe_sql("select a, b from t where a=1") is True

def test_sql_safety_check_invalid():
    service = TextToSQLService(settings=MagicMock(), session=MagicMock())
    assert service._is_safe_sql("UPDATE table SET a=1") is False
    assert service._is_safe_sql("DELETE FROM table") is False
    assert service._is_safe_sql("SELECT * FROM table; DROP TABLE another") is False
    assert service._is_safe_sql("INSERT INTO table VALUES (1)") is False
    assert service._is_safe_sql("TRUNCATE TABLE table") is False

def test_extract_sql_from_response():
    service = TextToSQLService(settings=MagicMock(), session=MagicMock())
    
    text1 = "Here is the SQL:\n```sql\nSELECT * FROM table\n```\nExplanation..."
    assert service._extract_sql_from_response(text1) == "SELECT * FROM table"
    
    text2 = "SELECT * FROM table"
    assert service._extract_sql_from_response(text2) == "SELECT * FROM table"

@pytest.mark.anyio
async def test_text_to_sql_execution_safety_exception():
    service = TextToSQLService(settings=MagicMock(), session=MagicMock())
    
    workspace_id = uuid4()
    dataset_ids = [uuid4()]
    
    # Executing an unsafe SQL should raise an exception
    with pytest.raises(SQLSafetyError):
        service._execute_sql("DROP TABLE users", workspace_id, dataset_ids)


@pytest.mark.anyio
async def test_text_to_sql_stream_generation_ollama_path_explains_without_gemini_client(monkeypatch):
    settings = MagicMock()
    settings.storage_local_path = "tmp"
    settings.gemini_api_key = ""
    settings.ollama_url = "http://ollama"
    service = TextToSQLService(settings=settings, session=MagicMock())

    dataset_asset = SimpleNamespace(id=uuid4(), title="sales")
    service._get_workspace_schema_context = AsyncMock(return_value=("Schema content", [dataset_asset]))
    service._generate_sql_ollama = AsyncMock(return_value="SELECT 1 AS a")
    service._execute_sql = MagicMock(
        return_value={"rows": [{"a": 1}], "row_count": 1, "sql_used": "SELECT 1 AS a"}
    )

    class FakeStreamResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self):
            return b""

        async def aiter_lines(self):
            yield json.dumps({"message": {"content": "Summary from Ollama"}})

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            return FakeStreamResponse()

    monkeypatch.setattr("api.app.services.sql.httpx.AsyncClient", FakeAsyncClient)

    tokens = []
    metas = []
    async for token, meta in service.stream_generation("llama3.2:1b", "show me sales", uuid4(), []):
        if token:
            tokens.append(token)
        if meta:
            metas.append(meta)

    assert any("SELECT 1 AS a" in token for token in tokens)
    assert any("Summary from Ollama" in token for token in tokens)
    assert metas[0]["provider"]["name"] == "ollama"
    assert metas[0]["sql_used"] == "SELECT 1 AS a"


@pytest.mark.anyio
async def test_generate_sql_gemini_retries_on_temporary_503():
    settings = MagicMock()
    settings.storage_local_path = "tmp"
    settings.gemini_api_key = "test-key"
    service = TextToSQLService(settings=settings, session=MagicMock())

    client = MagicMock()
    response = MagicMock()
    response.text = "```sql\nSELECT 1\n```"

    client.aio.models.generate_content = AsyncMock(
        side_effect=[
            RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand."),
            response,
        ]
    )

    sql = await service._generate_sql_gemini(client, "gemini-2.5-flash", "test", "schema")

    assert sql == "SELECT 1"
    assert client.aio.models.generate_content.await_count == 2


@pytest.mark.anyio
async def test_generate_sql_ollama_surfaces_error_detail(monkeypatch):
    settings = MagicMock()
    settings.storage_local_path = "tmp"
    settings.ollama_url = "http://ollama"
    service = TextToSQLService(settings=settings, session=MagicMock())

    class FakeResponse:
        status_code = 503

        async def aread(self):
            return b'{"error":"model overloaded"}'

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("api.app.services.sql.httpx.AsyncClient", FakeAsyncClient)

    with pytest.raises(RuntimeError, match="model overloaded"):
        await service._generate_sql_ollama("gemma", "show me sales", "schema")
