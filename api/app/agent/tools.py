import logging
from typing import Any
from uuid import UUID

from api.app.services.rag import RAGService
from api.app.services.sql import TextToSQLService

logger = logging.getLogger(__name__)


class AgentToolRegistry:
    def __init__(self, workspace_id: UUID, rag_service: RAGService, sql_service: TextToSQLService):
        self.workspace_id = workspace_id
        self.rag_service = rag_service
        self.sql_service = sql_service
        self._dataset_assets: list[Any] = []

    async def list_assets(self) -> dict[str, Any]:
        logger.info("Agent executing tool: list_assets()")
        return {"result": await self.sql_service.list_assets(self.workspace_id)}

    async def search_knowledge(self, query: str) -> dict[str, Any]:
        logger.info("Agent executing tool: search_knowledge(query=%s)", query)
        contexts = await self.rag_service.retrieve_context(self.workspace_id, query)
        if not contexts:
            return {"result": "No relevant documents found for this query."}

        return {
            "result": [
                {
                    "document": context.original_filename,
                    "asset_id": str(context.asset_id),
                    "chunk_id": str(context.chunk_id),
                    "content": context.content,
                    "source_page": context.source_page,
                }
                for context in contexts
            ]
        }

    async def get_knowledge_context(self, query: str) -> dict[str, Any]:
        logger.info("Agent executing tool: get_knowledge_context(query=%s)", query)
        return await self.search_knowledge(query)

    async def get_dataset_schema(self) -> dict[str, Any]:
        logger.info("Agent executing tool: get_dataset_schema()")
        schema_text, dataset_assets = await self.sql_service.get_workspace_schema_context(self.workspace_id)
        self._dataset_assets = dataset_assets
        if not schema_text:
            return {"result": "No dataset schemas available in this workspace."}
        return {"result": schema_text}

    async def get_dataset_profile(self) -> dict[str, Any]:
        logger.info("Agent executing tool: get_dataset_profile()")
        return {"result": await self.sql_service.get_dataset_profile(self.workspace_id)}

    async def preview_rows(self, table_name: str, limit: int = 5) -> dict[str, Any]:
        logger.info("Agent executing tool: preview_rows(table_name=%s, limit=%s)", table_name, limit)
        return await self.sql_service.preview_rows(self.workspace_id, table_name, limit)

    async def run_duckdb_sql(self, sql_query: str) -> dict[str, Any]:
        logger.info("Agent executing tool: run_duckdb_sql(sql_query=%s)", sql_query)
        if not self._dataset_assets:
            _, dataset_assets = await self.sql_service.get_workspace_schema_context(self.workspace_id)
            self._dataset_assets = dataset_assets

        if not self._dataset_assets:
            return {"error": "No datasets available to run SQL."}

        try:
            result = self.sql_service.execute_readonly_sql(sql_query, self.workspace_id, self._dataset_assets)
        except Exception as exc:
            return {"error": f"Unexpected error: {exc}"}

        if "error" in result:
            return {"error": f"SQL Execution Failed: {result['error']}"}

        rows = result.get("rows", [])
        row_count = result.get("row_count", 0)
        return {
            "total_rows": row_count,
            "data": rows[:50],
            "sql_used": result.get("sql_used"),
            "notice": "Results truncated to 50 rows if larger" if row_count > 50 else "",
        }

    async def ask_for_clarification(self, question: str) -> dict[str, Any]:
        logger.info("Agent executing tool: ask_for_clarification(question=%s)", question)
        return {"result": question or "Bạn cần cung cấp thêm ngữ cảnh để tôi chọn đúng công cụ."}
