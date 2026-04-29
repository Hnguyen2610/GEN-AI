import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, AsyncGenerator
from uuid import UUID

import httpx

import duckdb
from google import genai
from sqlalchemy import String, cast, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.app.core.config import Settings
from api.app.models.entities import Asset, AssetKind, ColumnProfile, DatasetSheet, DatasetStatus
from api.app.services.model_aliases import normalize_model_name

logger = logging.getLogger(__name__)


class SQLSafetyError(Exception):
    pass


class TextToSQLService:
    def __init__(self, settings: Settings, session: AsyncSession):
        self._settings = settings
        self._session = session
        self.duckdb_dir = Path(settings.storage_local_path) / "duckdb"

    def _is_safe_sql(self, sql: str) -> bool:
        normalized_sql = sql.upper().strip()
        statement_body = normalized_sql.rstrip(";")
        if ";" in statement_body:
            return False
        if not statement_body.startswith(("SELECT", "WITH")):
            return False

        forbidden_keywords = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "ALTER",
            "GRANT",
            "REVOKE",
            "CREATE",
            "REPLACE",
            "TRUNCATE",
            "MERGE",
            "EXEC",
            "EXECUTE",
            "COPY",
            "EXPORT",
            "ATTACH",
            "DETACH",
        ]
        return not any(re.search(rf"\b{keyword}\b", normalized_sql) for keyword in forbidden_keywords)

    async def list_assets(self, workspace_id: UUID) -> list[dict[str, Any]]:
        statement = (
            select(Asset)
            .where(
                Asset.workspace_id == workspace_id,
                cast(Asset.status, String) == DatasetStatus.ready.value,
            )
            .order_by(Asset.created_at.desc())
        )
        assets = (await self._session.scalars(statement)).all()
        return [
            {
                "id": str(asset.id),
                "kind": getattr(asset.kind, "value", asset.kind),
                "title": asset.title,
                "original_filename": asset.original_filename,
                "status": getattr(asset.status, "value", asset.status),
            }
            for asset in assets
        ]

    async def get_workspace_schema_context(self, workspace_id: UUID) -> tuple[str, list[Asset]]:
        return await self._get_workspace_schema_context(workspace_id)

    async def get_dataset_profile(self, workspace_id: UUID) -> dict[str, Any]:
        schema_text, assets = await self._get_workspace_schema_context(workspace_id)
        return {"schema": schema_text, "dataset_ids": [str(asset.id) for asset in assets]}

    async def preview_rows(self, workspace_id: UUID, table_name: str, limit: int = 5) -> dict[str, Any]:
        _, assets = await self._get_workspace_schema_context(workspace_id)
        if not assets:
            return {"error": "No datasets available."}

        safe_table = _quote_identifier(table_name)
        safe_limit = max(1, min(int(limit), 50))
        return self._execute_sql(f"SELECT * FROM {safe_table} LIMIT {safe_limit}", workspace_id, assets)

    def execute_readonly_sql(self, sql: str, workspace_id: UUID, assets: list[Asset]) -> dict[str, Any]:
        return self._execute_sql(sql, workspace_id, assets)

    async def _get_workspace_schema_context(self, workspace_id: UUID) -> tuple[str, list[Asset]]:
        asset_stmt = select(Asset).where(
            Asset.workspace_id == workspace_id,
            Asset.kind == AssetKind.dataset,
            cast(Asset.status, String) == DatasetStatus.ready.value,
        )
        assets = (await self._session.scalars(asset_stmt)).all()
        if not assets:
            return "", []

        schema_lines = ["Available Datasets and Schemas:"]
        for asset in assets:
            alias = _sanitize_alias(asset.title)
            schema_lines.extend(
                [
                    "",
                    f"Schema Name: {alias}",
                ]
            )
            schema_lines.extend(await self._dataset_schema_lines(asset.id, alias))

        return "\n".join(schema_lines), list(assets)

    async def _dataset_schema_lines(self, dataset_id: UUID, alias: str) -> list[str]:
        sheet_stmt = select(DatasetSheet).where(DatasetSheet.dataset_id == dataset_id)
        sheets = (await self._session.scalars(sheet_stmt)).all()
        lines: list[str] = []

        for sheet in sheets:
            lines.append(f"  Table: {alias}.{sheet.name}")
            col_stmt = select(ColumnProfile).where(
                ColumnProfile.dataset_id == dataset_id,
                ColumnProfile.sheet_name == sheet.name,
            )
            columns = (await self._session.scalars(col_stmt)).all()
            for column in columns:
                lines.append(
                    f"    - {column.column_name} ({column.data_type}): "
                    f"min={column.min_value}, max={column.max_value}"
                )

        return lines

    def _execute_sql(
        self, sql: str, workspace_id: UUID, datasets: list[Asset]
    ) -> dict[str, Any]:
        if not self._is_safe_sql(sql):
            raise SQLSafetyError("Generated SQL is not safe or is not a SELECT query.")

        conn = duckdb.connect()
        try:
            workspace_dir = self.duckdb_dir / str(workspace_id)
            attached = {}
            for asset in datasets:
                db_path = workspace_dir / f"{asset.id}.duckdb"
                if os.path.exists(db_path):
                    alias = _sanitize_alias(asset.title)
                    safe_path = str(db_path).replace("'", "''")
                    conn.execute(f"ATTACH '{safe_path}' AS \"{alias}\" (READ_ONLY)")
                    attached[asset.id] = alias

            result = conn.execute(sql).df()
            return {
                "columns": result.columns.tolist(),
                "rows": result.to_dict(orient="records"),
                "sql_used": sql,
                "row_count": len(result),
            }
        except Exception as exc:
            return {"error": str(exc), "sql_used": sql}
        finally:
            conn.close()

    def _extract_sql_from_response(self, text: str) -> str:
        match = re.search(r"```sql\s+(.*?)\s+```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    async def stream_generation(
        self,
        model_name: str,
        query: str,
        workspace_id: UUID,
        history: list[dict],
    ) -> AsyncGenerator[tuple[str, dict | None], None]:
        model_name = normalize_model_name(model_name)
        schema_context, dataset_assets = await self._get_workspace_schema_context(workspace_id)
        if not schema_context:
            yield "Không có dataset nào khả dụng để truy vấn SQL.", {
                "provider": {"name": "guardrail", "model": "none"},
                "route": "sql",
                "error": "No dataset available",
            }
            return

        is_gemini = model_name.startswith("gemini")
        provider_name = "gemini" if is_gemini else "ollama"
        if is_gemini and not self._settings.gemini_api_key:
            yield "GEMINI_API_KEY chưa được cấu hình nên không thể sinh truy vấn SQL.", {
                "provider": {"name": "guardrail", "model": "none"},
                "route": "sql",
                "error": "GEMINI_API_KEY is not configured",
            }
            return

        chat_model = model_name
        try:
            if is_gemini:
                client = genai.Client(api_key=self._settings.gemini_api_key)
                sql_query = await self._generate_sql_gemini(client, chat_model, query, schema_context)
            else:
                sql_query = await self._generate_sql_ollama(chat_model, query, schema_context)
        except Exception as e:
            logger.warning("SQL generation failed with %s: %s", model_name, e)
            detailed_error = _format_provider_error(e, provider_name, model_name, stage="sql_generation")
            yield f"Khong the sinh truy van SQL bang model {model_name}. {detailed_error}", {
                "provider": {"name": provider_name, "model": model_name},
                "route": "sql",
                "error": str(e),
                "error_detail": detailed_error,
                "error_stage": "sql_generation",
            }
            return

        if not sql_query:
            yield "Tôi không thể tạo được câu truy vấn SQL phù hợp cho câu hỏi này.", {
                "provider": {"name": provider_name, "model": chat_model},
                "route": "sql",
                "error": "LLM failed to generate SQL",
            }
            return

        try:
            result = self._execute_sql(sql_query, workspace_id, dataset_assets)
        except SQLSafetyError as exc:
            yield f"Bảo mật: truy vấn tạo ra không hợp lệ ({exc}).", {
                "provider": {"name": provider_name, "model": chat_model},
                "route": "sql",
                "error": str(exc),
            }
            return

        if "error" in result:
            detailed_error = f"SQL execution failed: {result['error']}"
            yield f"Truy van SQL that bai. {detailed_error}", {
                "provider": {"name": provider_name, "model": chat_model},
                "route": "sql",
                "sql_used": result.get("sql_used"),
                "error": result["error"],
                "error_detail": detailed_error,
                "error_stage": "sql_execution",
            }
            return

        async for item in self._explain_result(provider_name, chat_model, query, result):
            yield item

    async def _generate_sql_gemini(self, client: genai.Client, model_name: str, query: str, schema_context: str) -> str:
        prompt = f"""Bạn là chuyên gia SQL. Hãy chuyển câu hỏi thành một truy vấn DuckDB an toàn.
Chỉ được dùng SELECT hoặc WITH. Không tạo, sửa, xoá, export dữ liệu.
BẮT BUỘC sử dụng Schema Name kèm theo tên bảng (ví dụ: schema_name.table_name) trong câu SELECT.
Nếu không thể viết SQL phù hợp, trả về chuỗi rỗng.
Trả về duy nhất SQL trong block ```sql ... ```.

{schema_context}

Câu hỏi người dùng: {query}
"""
        response = await self._call_gemini_with_retry(
            lambda: client.aio.models.generate_content(model=model_name, contents=prompt)
        )
        return self._extract_sql_from_response(response.text or "")

    async def _generate_sql_ollama(self, model_name: str, query: str, schema_context: str) -> str:
        prompt = f"""Bạn là chuyên gia SQL. Hãy chuyển câu hỏi thành một truy vấn DuckDB an toàn.
CHỈ TRẢ VỀ DUY NHẤT câu lệnh SQL trong block ```sql ... ```. 
Không giải thích gì thêm.
BẮT BUỘC sử dụng Schema Name (Ví dụ: `schema_name.table_name`).

{schema_context}

Câu hỏi người dùng: {query}
SQL:"""
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_ctx": self._settings.ollama_num_ctx},
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(f"{self._settings.ollama_url}/api/chat", json=payload)
            await _raise_ollama_error(res, model_name)
            content = res.json()["message"]["content"]
            return self._extract_sql_from_response(content)

    async def _explain_result(
        self,
        provider_name: str,
        model_name: str,
        query: str,
        result: dict[str, Any],
    ) -> AsyncGenerator[tuple[str, dict | None], None]:
        rows = result["rows"]
        sql_used = result["sql_used"]
        data_str = str(rows[:10])
        if len(rows) > 10:
            data_str += f"\n... (and {len(rows) - 10} more rows)"

        explanation_prompt = f"""Bạn vừa chạy một truy vấn SQL và có kết quả dưới đây.
Hãy giải thích kết quả bằng tiếng Việt, ngắn gọn và dễ hiểu.

Câu hỏi gốc: {query}
Câu truy vấn: {sql_used}
Kết quả tối đa 10 dòng đầu:
{data_str}
"""
        yield f"**Truy vấn SQL đã chạy:**\n```sql\n{sql_used}\n```\n\n**Kết quả phân tích:**\n", None

        try:
            if provider_name == "gemini":
                client = genai.Client(api_key=self._settings.gemini_api_key)
                response = await self._call_gemini_with_retry(
                    lambda: client.aio.models.generate_content_stream(
                        model=model_name,
                        contents=explanation_prompt,
                    )
                )
                async for chunk in response:
                    if chunk.text:
                        yield chunk.text, None
            else:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": explanation_prompt}],
                    "stream": True,
                    "options": {"num_ctx": self._settings.ollama_num_ctx},
                }
                async with httpx.AsyncClient(timeout=60.0) as ollama_client:
                    async with ollama_client.stream(
                        "POST", f"{self._settings.ollama_url}/api/chat", json=payload
                    ) as response:
                        await _raise_ollama_error(response, model_name)
                        async for line in response.aiter_lines():
                            if not line: continue
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                yield chunk["message"]["content"], None
        except Exception:
            logger.exception("SQL result explanation failed")
            yield "Da chay SQL thanh cong nhung khong the sinh phan giai thich tu nhien.", None

        yield "", {
            "provider": {"name": provider_name, "model": model_name},
            "route": "sql",
            "sql_used": sql_used,
            "row_count": result["row_count"],
            "data_preview": rows[:5],
            "error": None,
        }

    async def _call_gemini_with_retry(self, operation, retries: int = 2):
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return await operation()
            except Exception as exc:
                last_error = exc
                if not _is_retryable_gemini_error(exc) or attempt == retries:
                    raise
                await asyncio.sleep(0.75 * (attempt + 1))

        raise last_error or RuntimeError("Gemini call failed")


def _quote_identifier(identifier: str) -> str:
    parts = identifier.split(".")
    if not parts or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in parts):
        raise SQLSafetyError("Invalid table identifier.")
    return ".".join(f'"{part}"' for part in parts)


def _sanitize_alias(name: str) -> str:
    # DuckDB aliases should be clean. Remove spaces/dashes and extension.
    base = os.path.splitext(name)[0]
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", base).lower().strip("_")
    return sanitized if sanitized else "dataset"


def _is_retryable_gemini_error(exc: Exception) -> bool:
    detail = str(exc).upper()
    return "503" in detail or "UNAVAILABLE" in detail or "HIGH DEMAND" in detail


def _format_provider_error(exc: Exception, provider_name: str, model_name: str, stage: str) -> str:
    detail = str(exc).strip() or "Unknown provider error."

    if provider_name == "gemini" and _is_retryable_gemini_error(exc):
        return (
            f"Gemini provider returned a temporary overload during {stage} after retries. "
            f"Model: {model_name}. Detail: {detail}"
        )

    if provider_name == "ollama":
        return f"Ollama failed during {stage}. Model: {model_name}. Detail: {detail}"

    return f"Provider {provider_name} failed during {stage}. Model: {model_name}. Detail: {detail}"


async def _raise_ollama_error(response: httpx.Response, model_name: str) -> None:
    if response.status_code < 400:
        return

    body = await response.aread()
    detail = body.decode("utf-8", errors="replace").strip() or "No response body returned by Ollama."
    try:
        parsed = json.loads(detail)
        detail = parsed.get("error", detail)
    except json.JSONDecodeError:
        pass
    raise RuntimeError(f"Ollama model {model_name} failed: {detail}")
