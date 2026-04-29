import logging
from typing import Any, AsyncGenerator
from uuid import UUID

from google import genai
from google.genai import types

from api.app.agent.tools import AgentToolRegistry
from api.app.core.config import Settings
from api.app.services.model_aliases import DEFAULT_GEMINI_MODEL, normalize_model_name

logger = logging.getLogger(__name__)


class PEVAgentService:
    def __init__(self, settings: Settings):
        self._settings = settings

    def _get_tool_declarations(self) -> list[types.Tool]:
        declarations = [
            _tool("list_assets", "List all ready assets in the workspace.", {}),
            _tool(
                "search_knowledge",
                "Search text documents for policies, rules, procedures, definitions, or semantic context.",
                {"query": ("STRING", "Search query.")},
                ["query"],
            ),
            _tool(
                "get_knowledge_context",
                "Get grounded context from knowledge documents for a specific query.",
                {"query": ("STRING", "Context query.")},
                ["query"],
            ),
            _tool("get_dataset_schema", "Get schemas of all CSV/Excel datasets.", {}),
            _tool("get_dataset_profile", "Get dataset profile and schema summary.", {}),
            _tool(
                "preview_rows",
                "Preview rows from a dataset table before writing SQL.",
                {
                    "table_name": ("STRING", "DuckDB table name."),
                    "limit": ("INTEGER", "Maximum rows, capped by the backend."),
                },
                ["table_name"],
            ),
            _tool(
                "run_duckdb_sql",
                "Run a safe read-only SELECT/WITH DuckDB query against available datasets.",
                {"sql_query": ("STRING", "Safe SELECT/WITH SQL query.")},
                ["sql_query"],
            ),
            _tool(
                "ask_for_clarification",
                "Ask the user for missing context when no safe plan can be executed.",
                {"question": ("STRING", "Clarification question.")},
                ["question"],
            ),
        ]
        return [types.Tool(function_declarations=declarations)]

    async def stream_response(
        self,
        workspace_id: UUID,
        query: str,
        history: list[dict[str, str]],
        model_name: str,
        registry: AgentToolRegistry,
    ) -> AsyncGenerator[tuple[str, dict | None], None]:
        if not self._settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured. Agent PEV requires Gemini tool calling.")

        client = genai.Client(api_key=self._settings.gemini_api_key)
        model_name = normalize_model_name(model_name)
        chat_model = model_name if model_name.startswith("gemini") else DEFAULT_GEMINI_MODEL
        messages = _build_messages(query, history)
        config = types.GenerateContentConfig(tools=self._get_tool_declarations(), temperature=0.1)

        max_steps = 5
        traces: list[dict[str, Any]] = []

        for step in range(1, max_steps + 1):
            response = await client.aio.models.generate_content(
                model=chat_model,
                contents=messages,
                config=config,
            )
            if response.candidates and response.candidates[0].content:
                messages.append(response.candidates[0].content)

            function_calls = _extract_function_calls(response)
            if not function_calls:
                final_text = response.text or "Không thể đưa ra câu trả lời."
                yield final_text, None
                yield "", {
                    "route": "agent",
                    "steps_taken": step,
                    "agent_traces": traces,
                    "verification": _verify_traces(traces),
                }
                return

            tool_parts = []
            for function_call in function_calls:
                result = await self._execute_tool(registry, function_call)
                traces.append(
                    {
                        "step": step,
                        "tool": function_call.name,
                        "args": dict(function_call.args or {}),
                        "result": "error" if "error" in result else "success",
                    }
                )
                tool_parts.append(
                    types.Part.from_function_response(
                        name=function_call.name,
                        response=result,
                    )
                )

            messages.append(types.Content(role="user", parts=tool_parts))

        fallback = "Tôi đã thử nhiều bước nhưng chưa xác minh được câu trả lời phù hợp trong dữ liệu của bạn."
        yield fallback, None
        yield "", {
            "route": "agent_aborted",
            "steps_taken": max_steps,
            "agent_traces": traces,
            "verification": _verify_traces(traces),
        }

    async def _execute_tool(self, registry: AgentToolRegistry, function_call: Any) -> dict[str, Any]:
        args = dict(function_call.args or {})
        name = function_call.name
        logger.info("Agent requested tool: %s with args %s", name, args)

        try:
            if name == "list_assets":
                return await registry.list_assets()
            if name == "search_knowledge":
                return await registry.search_knowledge(args.get("query", ""))
            if name == "get_knowledge_context":
                return await registry.get_knowledge_context(args.get("query", ""))
            if name == "get_dataset_schema":
                return await registry.get_dataset_schema()
            if name == "get_dataset_profile":
                return await registry.get_dataset_profile()
            if name == "preview_rows":
                return await registry.preview_rows(args.get("table_name", ""), args.get("limit", 5))
            if name == "run_duckdb_sql":
                return await registry.run_duckdb_sql(args.get("sql_query", ""))
            if name == "ask_for_clarification":
                return await registry.ask_for_clarification(args.get("question", ""))
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return {"error": str(exc)}

        return {"error": f"Tool not found: {name}"}


def _build_messages(query: str, history: list[dict[str, str]]) -> list[types.Content]:
    system_prompt = """Bạn là AI Agent theo pattern PEV: Plan -> Execute -> Verify.
Bạn có thể tra cứu tài liệu bằng RAG và phân tích CSV/Excel bằng DuckDB SQL.
Luôn lập kế hoạch ngắn trong suy luận nội bộ, gọi công cụ cần thiết, rồi xác minh kết quả trước khi trả lời.
Không bịa dữ liệu. Nếu thiếu thông tin hoặc không có tool phù hợp, hãy hỏi rõ lại.
Trả lời cuối cùng bằng tiếng Việt, ngắn gọn, có căn cứ từ tool đã dùng."""

    messages = [
        types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)]),
        types.Content(role="model", parts=[types.Part.from_text(text="Đã hiểu. Tôi sẽ dùng công cụ và xác minh trước khi trả lời.")]),
    ]
    for message in history:
        role = "user" if message["role"] == "user" else "model"
        messages.append(types.Content(role=role, parts=[types.Part.from_text(text=message["content"])]))
    messages.append(types.Content(role="user", parts=[types.Part.from_text(text=query)]))
    return messages


def _extract_function_calls(response: Any) -> list[Any]:
    if not response.candidates or not response.candidates[0].content:
        return []

    calls = []
    for part in response.candidates[0].content.parts or []:
        if part.function_call:
            calls.append(part.function_call)
    return calls


def _tool(
    name: str,
    description: str,
    properties: dict[str, tuple[str, str]],
    required: list[str] | None = None,
) -> types.FunctionDeclaration:
    type_map = {
        "STRING": types.Type.STRING,
        "INTEGER": types.Type.INTEGER,
    }
    schema_properties = {
        key: types.Schema(type=type_map[value_type], description=field_description)
        for key, (value_type, field_description) in properties.items()
    }
    return types.FunctionDeclaration(
        name=name,
        description=description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties=schema_properties,
            required=required or [],
        ),
    )


def _verify_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    failed_tools = [trace for trace in traces if trace["result"] == "error"]
    return {
        "status": "failed" if failed_tools else "passed",
        "tool_calls": len(traces),
        "failed_tools": failed_tools,
    }
