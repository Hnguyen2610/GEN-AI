import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from api.app.agent.pev import (
    PEVAgentService,
    ToolRun,
    _build_primary_sources,
    _run_local_logic_checks,
)
from api.app.agent.tools import AgentToolRegistry


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode("utf-8")

    def json(self):
        return self._payload

    async def aread(self):
        return self.content


class FakeAsyncClient:
    responses: list[FakeResponse] = []

    def __init__(self, *args, **kwargs):
        self._responses = type(self).responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        if not self._responses:
            raise AssertionError("No fake Groq responses remaining.")
        return self._responses.pop(0)


def tool_call_response(name: str, arguments: dict, call_id: str = "call_1") -> FakeResponse:
    return FakeResponse(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )


def final_answer_response(content: str) -> FakeResponse:
    return FakeResponse({"choices": [{"message": {"role": "assistant", "content": content}}]})


def verifier_response(status: str, reason: str) -> FakeResponse:
    return FakeResponse(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "status": status,
                                "reason": reason,
                                "checks": [
                                    {
                                        "name": "semantic_grounding",
                                        "status": "passed" if status == "passed" else "failed",
                                        "detail": reason,
                                    }
                                ],
                                "safe_response": "Verifier fallback response.",
                            }
                        )
                    }
                }
            ]
        }
    )


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.groq_api_key = "test_key"
    settings.agent_max_steps = 6
    settings.agent_verification_retries = 1
    settings.agent_enable_logic_verification = True
    return settings


@pytest.fixture
def mock_registry():
    registry = AsyncMock(spec=AgentToolRegistry)
    registry.search_knowledge.return_value = {
        "result": [
            {
                "document": "policy.md",
                "asset_id": str(uuid4()),
                "chunk_id": str(uuid4()),
                "content": "Remote work requires manager approval.",
                "source_page": None,
            }
        ]
    }
    registry.run_duckdb_sql.return_value = {
        "total_rows": 1,
        "data": [{"total": 42}],
        "sql_used": "SELECT 42 AS total",
        "dataset_sources": [
            {
                "kind": "dataset",
                "asset_id": str(uuid4()),
                "title": "sample_rag_test",
                "original_filename": "sample_rag_test.xlsx",
                "schema_name": "sample_rag_test",
            }
        ],
    }
    return registry


async def collect_stream(stream):
    tokens = []
    metas = []
    async for token, meta in stream:
        if token:
            tokens.append(token)
        if meta:
            metas.append(meta)
    return tokens, metas


@pytest.mark.anyio
async def test_agent_verifies_grounded_knowledge_answer(monkeypatch, mock_settings, mock_registry):
    FakeAsyncClient.responses = [
        tool_call_response("search_knowledge", {"query": "remote work policy"}),
        final_answer_response("Remote work requires manager approval."),
        verifier_response("passed", "The answer matches the cited policy."),
    ]
    monkeypatch.setattr("api.app.agent.pev.httpx.AsyncClient", FakeAsyncClient)

    service = PEVAgentService(settings=mock_settings)
    tokens, metas = await collect_stream(
        service.stream_response(
            workspace_id=uuid4(),
            query="What is the remote work policy?",
            history=[],
            model_name="llama-3.3-70b-versatile",
            registry=mock_registry,
        )
    )

    assert tokens[-1] == "Remote work requires manager approval."
    assert metas[0]["verification"]["status"] == "passed"
    assert metas[0]["verification"]["attempts"] == 0
    assert len(metas[0]["citations"]) == 1
    mock_registry.search_knowledge.assert_awaited_once()


@pytest.mark.anyio
async def test_agent_self_corrects_after_local_sql_verification_failure(
    monkeypatch, mock_settings, mock_registry
):
    FakeAsyncClient.responses = [
        tool_call_response("run_duckdb_sql", {"sql_query": "SELECT 42 AS total"}),
        final_answer_response("Tong doanh thu la 100."),
        final_answer_response("Tong doanh thu la 42."),
        verifier_response("passed", "The revised answer now matches the SQL output."),
    ]
    monkeypatch.setattr("api.app.agent.pev.httpx.AsyncClient", FakeAsyncClient)

    service = PEVAgentService(settings=mock_settings)
    tokens, metas = await collect_stream(
        service.stream_response(
            workspace_id=uuid4(),
            query="Tong doanh thu la bao nhieu?",
            history=[],
            model_name="llama-3.3-70b-versatile",
            registry=mock_registry,
        )
    )

    assert tokens[-1] == "Tong doanh thu la 42."
    assert metas[0]["verification"]["status"] == "passed"
    assert metas[0]["verification"]["attempts"] == 1
    assert metas[0]["dataset_sources"][0]["original_filename"] == "sample_rag_test.xlsx"
    assert metas[0]["primary_sources"][0]["kind"] == "dataset"
    assert any(
        trace["tool"] == "logic_verification" and trace["result"] == "needs_revision"
        for trace in metas[0]["agent_traces"]
    )
    mock_registry.run_duckdb_sql.assert_awaited_once()


def test_local_logic_checks_require_successful_tool_evidence():
    result = _run_local_logic_checks(
        query="Any answer?",
        answer="This should not pass.",
        tool_runs=[],
        citations=[],
    )

    assert result["status"] == "insufficient_evidence"
    assert "successful tool evidence" in result["reason"]


def test_local_logic_checks_catch_sql_number_mismatch():
    tool_runs = [
        ToolRun(
            step=1,
            tool="run_duckdb_sql",
            args={"sql_query": "SELECT 42 AS total"},
            result="success",
            output={"total_rows": 1, "data": [{"total": 42}], "sql_used": "SELECT 42 AS total"},
        )
    ]

    result = _run_local_logic_checks(
        query="Tong doanh thu la bao nhieu?",
        answer="Tong doanh thu la 100.",
        tool_runs=tool_runs,
        citations=[],
    )

    assert result["status"] == "needs_revision"
    assert "Numeric claims" in result["reason"]


def test_build_primary_sources_prefers_best_document_match_from_latest_knowledge_run():
    tool_runs = [
        ToolRun(
            step=1,
            tool="search_knowledge",
            args={"query": "premium members"},
            result="success",
            output={
                "result": [
                    {
                        "document": "docs1.docx",
                        "asset_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "content": "Company overview and office rules.",
                        "source_page": None,
                    },
                    {
                        "document": "docs2.docx",
                        "asset_id": "doc-2",
                        "chunk_id": "chunk-2",
                        "content": "Premium members include Nguyen A, Tran B, Le C, Hoang E.",
                        "source_page": None,
                    },
                ]
            },
        )
    ]

    primary_sources = _build_primary_sources(
        query="Nhung thanh vien premium va standard gom nhung ai?",
        answer="Nhung thanh vien premium va standard gom Nguyen A, Tran B, Le C, Hoang E.",
        tool_runs=tool_runs,
        citations=[
            {"kind": "document", "asset_id": "doc-1", "original_filename": "docs1.docx", "quote": "Company overview."},
            {"kind": "document", "asset_id": "doc-2", "original_filename": "docs2.docx", "quote": "Premium members include Nguyen A, Tran B, Le C, Hoang E."},
        ],
        dataset_sources=[],
    )

    assert [source["original_filename"] for source in primary_sources] == ["docs2.docx"]


def test_build_primary_sources_prefers_latest_sql_run_sources_only():
    tool_runs = [
        ToolRun(
            step=1,
            tool="run_duckdb_sql",
            args={"sql_query": "SELECT * FROM dataset_one.customers"},
            result="success",
            output={
                "total_rows": 1,
                "data": [{"name": "Old source"}],
                "sql_used": "SELECT * FROM dataset_one.customers",
                "dataset_sources": [
                    {
                        "kind": "dataset",
                        "asset_id": "ds-1",
                        "title": "dataset_one",
                        "original_filename": "dataset_one.xlsx",
                        "schema_name": "dataset_one",
                    }
                ],
            },
        ),
        ToolRun(
            step=2,
            tool="run_duckdb_sql",
            args={"sql_query": "SELECT * FROM dataset_two.customers"},
            result="success",
            output={
                "total_rows": 1,
                "data": [{"name": "Right source"}],
                "sql_used": "SELECT * FROM dataset_two.customers",
                "dataset_sources": [
                    {
                        "kind": "dataset",
                        "asset_id": "ds-2",
                        "title": "dataset_two",
                        "original_filename": "dataset_two.xlsx",
                        "schema_name": "dataset_two",
                    }
                ],
            },
        ),
    ]

    primary_sources = _build_primary_sources(
        query="Lay danh sach khach hang",
        answer="Danh sach khach hang lay tu dataset hai.",
        tool_runs=tool_runs,
        citations=[],
        dataset_sources=[
            {"kind": "dataset", "asset_id": "ds-1", "title": "dataset_one", "original_filename": "dataset_one.xlsx", "schema_name": "dataset_one"},
            {"kind": "dataset", "asset_id": "ds-2", "title": "dataset_two", "original_filename": "dataset_two.xlsx", "schema_name": "dataset_two"},
        ],
    )

    assert [source["original_filename"] for source in primary_sources] == ["dataset_two.xlsx"]
