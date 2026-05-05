import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, AsyncGenerator
from uuid import UUID

import httpx

from api.app.agent.tools import AgentToolRegistry
from api.app.core.config import Settings
from api.app.services.model_aliases import normalize_model_name

logger = logging.getLogger(__name__)


SAFE_VERIFICATION_RESPONSE = (
    "Tôi chưa thể xác minh câu trả lời này từ dữ liệu hiện có. "
    "Hãy chỉ rõ hơn tài liệu,bảng dữ liệu hay yêu cầu tính toán cần dùng."
)


@dataclass
class ToolRun:
    step: int
    tool: str
    args: dict[str, Any]
    result: str
    output: Any


class PEVAgentService:
    def __init__(self, settings: Settings):
        self._settings = settings

    def _get_tool_declarations(self) -> list[dict[str, Any]]:
        return [
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

    async def stream_response(
        self,
        workspace_id: UUID,
        query: str,
        history: list[dict[str, str]],
        model_name: str,
        registry: AgentToolRegistry,
    ) -> AsyncGenerator[tuple[str, dict | None], None]:
        if not self._settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured. Agent PEV requires Groq tool calling.")

        api_model_name = normalize_model_name(model_name)
        if "/" in api_model_name:
            api_model_name = api_model_name.split("/")[-1]

        messages = _build_messages(query, history)
        tools = self._get_tool_declarations()
        headers = {
            "Authorization": f"Bearer {self._settings.groq_api_key}",
            "Content-Type": "application/json",
        }

        max_steps = max(1, int(getattr(self._settings, "agent_max_steps", 8)))
        max_revision_attempts = max(0, int(getattr(self._settings, "agent_verification_retries", 1)))
        enable_logic_verification = bool(
            getattr(self._settings, "agent_enable_logic_verification", True)
        )

        traces: list[dict[str, Any]] = []
        tool_runs: list[ToolRun] = []
        citations: list[dict[str, Any]] = []
        seen_filenames: set[str] = set()
        dataset_sources: list[dict[str, Any]] = []
        seen_dataset_sources: set[str] = set()
        revision_attempts = 0

        async with httpx.AsyncClient(timeout=60.0) as client:
            for step in range(1, max_steps + 1):
                payload = {
                    "model": api_model_name,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": 0.1,
                }

                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )

                response_data = None
                if res.status_code == 400:
                    response_data = _recover_failed_tool_use(res, step)

                if res.status_code >= 400 and response_data is None:
                    body = await res.aread() if hasattr(res, "aread") else res.content
                    raise RuntimeError(
                        f"Groq API error ({res.status_code}): {body.decode('utf-8', errors='ignore')}"
                    )
                if response_data is None:
                    response_data = res.json()

                choice = response_data.get("choices", [{}])[0]
                message = choice.get("message", {})

                if message.get("content") or message.get("tool_calls"):
                    assistant_message = {"role": "assistant"}
                    if message.get("content"):
                        assistant_message["content"] = message["content"]
                    if message.get("tool_calls"):
                        assistant_message["tool_calls"] = message["tool_calls"]
                    messages.append(assistant_message)

                tool_calls = message.get("tool_calls")
                if not tool_calls:
                    final_text = message.get("content", "Khong the dua ra cau tra loi.")
                    verification = (
                        await self._verify_final_answer(
                            client=client,
                            api_model_name=api_model_name,
                            query=query,
                            answer=final_text,
                            tool_runs=tool_runs,
                            citations=citations,
                        )
                        if enable_logic_verification
                        else _verification_result(
                            status="skipped",
                            reason="Logic verification disabled by configuration.",
                            checks=[],
                            method="disabled",
                            safe_response=SAFE_VERIFICATION_RESPONSE,
                        )
                    )
                    verification["attempts"] = revision_attempts

                    if (
                        verification["status"] == "needs_revision"
                        and revision_attempts < max_revision_attempts
                    ):
                        revision_attempts += 1
                        verification["attempts"] = revision_attempts
                        traces.append(
                            {
                                "step": step,
                                "tool": "logic_verification",
                                "args": {"attempt": revision_attempts},
                                "result": "needs_revision",
                                "reason": verification["reason"],
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": _build_revision_feedback(query, final_text, verification),
                            }
                        )
                        continue

                    traces.append(
                        {
                            "step": step,
                            "tool": "logic_verification",
                            "args": {"attempt": revision_attempts},
                            "result": verification["status"],
                            "reason": verification["reason"],
                        }
                    )

                    if verification["status"] != "passed":
                        final_text = verification.get("safe_response") or SAFE_VERIFICATION_RESPONSE

                    primary_sources = _build_primary_sources(
                        query=query,
                        answer=final_text,
                        tool_runs=tool_runs,
                        citations=citations,
                        dataset_sources=dataset_sources,
                    )
                    yield final_text, None
                    yield "", {
                        "route": "agent",
                        "steps_taken": step,
                        "agent_traces": traces,
                        "citations": citations,
                        "dataset_sources": dataset_sources,
                        "primary_sources": primary_sources,
                        "verification": verification,
                    }
                    return

                for call in tool_calls:
                    function_call = call.get("function", {})
                    call_name = function_call.get("name")
                    call_id = call.get("id")

                    try:
                        args_dict = json.loads(function_call.get("arguments", "{}"))
                    except BaseException:
                        args_dict = {}

                    yield f"\n> AI suy luan: su dung cong cu `{call_name}`...\n\n", None

                    result = await self._execute_tool(registry, call_name, args_dict)
                    result_status = (
                        "error" if isinstance(result, dict) and "error" in result else "success"
                    )

                    tool_runs.append(
                        ToolRun(
                            step=step,
                            tool=call_name,
                            args=args_dict,
                            result=result_status,
                            output=result,
                        )
                    )

                    _collect_citations(call_name, result, citations, seen_filenames)
                    _collect_dataset_sources(
                        call_name,
                        result,
                        dataset_sources,
                        seen_dataset_sources,
                    )

                    traces.append(
                        {
                            "step": step,
                            "tool": call_name,
                            "args": args_dict,
                            "result": result_status,
                        }
                    )

                    if isinstance(result, str):
                        result_str = result
                    else:
                        result_str = json.dumps(result, ensure_ascii=False, default=str)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": call_name,
                            "content": result_str,
                        }
                    )

        fallback = (
            "Tôi đã thử nhiều bước nhưng chưa xác minh được câu trả lời "
            "trong dữ liệu của bạn."
        )
        yield fallback, None
        yield "", {
            "route": "agent_aborted",
            "steps_taken": max_steps,
            "agent_traces": traces,
            "citations": citations,
            "dataset_sources": dataset_sources,
            "primary_sources": _build_primary_sources(
                query=query,
                answer=fallback,
                tool_runs=tool_runs,
                citations=citations,
                dataset_sources=dataset_sources,
            ),
            "verification": _verification_result(
                status="failed",
                reason="Agent exhausted max steps before producing a verified answer.",
                checks=[],
                method="step_budget",
                safe_response=SAFE_VERIFICATION_RESPONSE,
            ),
        }

    async def _execute_tool(self, registry: AgentToolRegistry, name: str, args: dict) -> Any:
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

    async def _verify_final_answer(
        self,
        client: httpx.AsyncClient,
        api_model_name: str,
        query: str,
        answer: str,
        tool_runs: list[ToolRun],
        citations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        local_result = _run_local_logic_checks(query, answer, tool_runs, citations)
        if local_result["status"] != "passed":
            return local_result

        semantic_result = await self._semantic_verify_final_answer(
            client=client,
            api_model_name=api_model_name,
            query=query,
            answer=answer,
            tool_runs=tool_runs,
            citations=citations,
            local_checks=local_result["checks"],
        )
        if semantic_result["status"] == "skipped":
            return semantic_result

        return {
            **semantic_result,
            "checks": [*local_result["checks"], *semantic_result.get("checks", [])],
        }

    async def _semantic_verify_final_answer(
        self,
        client: httpx.AsyncClient,
        api_model_name: str,
        query: str,
        answer: str,
        tool_runs: list[ToolRun],
        citations: list[dict[str, Any]],
        local_checks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        verification_prompt = _build_semantic_verifier_prompt(
            query=query,
            answer=answer,
            tool_runs=tool_runs,
            citations=citations,
            local_checks=local_checks,
        )
        payload = {
            "model": api_model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a logic verification layer for an internal agent. "
                        "Judge the candidate answer only against the provided tool outputs. "
                        "Never assume facts beyond the evidence."
                    ),
                },
                {"role": "user", "content": verification_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }

        try:
            res = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
            )
            if res.status_code >= 400:
                body = await res.aread()
                raise RuntimeError(
                    f"Groq verifier error ({res.status_code}): {body.decode('utf-8', errors='ignore')}"
                )

            response_data = res.json()
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            parsed = json.loads(content)
            status = parsed.get("status", "needs_revision")
            if status not in {"passed", "needs_revision", "insufficient_evidence"}:
                status = "needs_revision"

            checks = parsed.get("checks")
            if not isinstance(checks, list):
                checks = []

            return _verification_result(
                status=status,
                reason=str(parsed.get("reason", "Semantic verifier completed.")),
                checks=checks,
                method="semantic_verifier",
                safe_response=str(parsed.get("safe_response") or SAFE_VERIFICATION_RESPONSE),
            )
        except Exception as exc:
            logger.warning("Semantic verifier unavailable, falling back to local checks: %s", exc)
            return _verification_result(
                status="skipped",
                reason=f"Semantic verifier skipped: {exc}",
                checks=[],
                method="local_fallback",
                safe_response=SAFE_VERIFICATION_RESPONSE,
            )


def _build_messages(query: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
    system_prompt = (
        "Ban la AI Agent cua Cong ty. BAT BUOC tra loi moi cau hoi dua tren tai lieu "
        "(search_knowledge) hoac bang du lieu (run_duckdb_sql). Khong duoc bia hoac "
        "tra loi bang kien thuc ben ngoai. Neu thieu du lieu, phai noi ro hoac hoi lai."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "assistant",
            "content": "Da hieu. Toi se chi dung cong cu va tai lieu noi bo truoc khi tra loi.",
        },
    ]
    for message in history:
        role = "user" if message["role"] == "user" else "assistant"
        messages.append({"role": role, "content": message.get("content", "")})

    messages.append({"role": "user", "content": query})
    return messages


def _run_local_logic_checks(
    query: str,
    answer: str,
    tool_runs: list[ToolRun],
    citations: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    clean_answer = answer.strip()
    if not clean_answer:
        return _verification_result(
            status="needs_revision",
            reason="Final answer was empty.",
            checks=[_check("answer_presence", "failed", "Agent returned an empty answer.")],
            method="local_checks",
            safe_response=SAFE_VERIFICATION_RESPONSE,
        )

    successful_runs = [run for run in tool_runs if run.result == "success"]
    failed_runs = [run for run in tool_runs if run.result == "error"]

    if not successful_runs:
        return _verification_result(
            status="insufficient_evidence",
            reason="Agent attempted to answer without any successful tool evidence.",
            checks=[_check("tool_evidence", "failed", "No successful tool outputs were available.")],
            method="local_checks",
            safe_response=SAFE_VERIFICATION_RESPONSE,
        )

    checks.append(
        _check(
            "tool_evidence",
            "passed",
            f"Agent collected {len(successful_runs)} successful tool output(s).",
        )
    )

    if failed_runs:
        details = ", ".join(run.tool for run in failed_runs)
        return _verification_result(
            status="needs_revision",
            reason=f"One or more tools failed before the final answer: {details}.",
            checks=[
                *checks,
                _check("tool_failures", "failed", f"Failed tools: {details}."),
            ],
            method="local_checks",
            safe_response=SAFE_VERIFICATION_RESPONSE,
        )

    knowledge_runs = [
        run for run in successful_runs if run.tool in {"search_knowledge", "get_knowledge_context"}
    ]
    sql_runs = [run for run in successful_runs if run.tool == "run_duckdb_sql"]

    if knowledge_runs and not citations:
        return _verification_result(
            status="needs_revision",
            reason="Knowledge answer is missing supporting citations.",
            checks=[
                *checks,
                _check("knowledge_citations", "failed", "Knowledge tools ran without producing citations."),
            ],
            method="local_checks",
            safe_response=SAFE_VERIFICATION_RESPONSE,
        )

    if knowledge_runs:
        checks.append(
            _check(
                "knowledge_citations",
                "passed",
                f"Knowledge evidence attached through {len(citations)} citation(s).",
            )
        )
        if _knowledge_evidence_is_empty(knowledge_runs) and not _looks_like_safe_answer(clean_answer):
            return _verification_result(
                status="needs_revision",
                reason="Knowledge retrieval returned no evidence, but the answer remained confident.",
                checks=[
                    *checks,
                    _check(
                        "knowledge_grounding",
                        "failed",
                        "No relevant document context was found for the final answer.",
                    ),
                ],
                method="local_checks",
                safe_response=SAFE_VERIFICATION_RESPONSE,
            )

    sql_check = _check_sql_alignment(clean_answer, sql_runs)
    if sql_check is not None:
        checks.append(sql_check)
        if sql_check["status"] == "failed":
            return _verification_result(
                status="needs_revision",
                reason=sql_check["detail"],
                checks=checks,
                method="local_checks",
                safe_response=SAFE_VERIFICATION_RESPONSE,
            )

    return _verification_result(
        status="passed",
        reason="Local evidence checks passed.",
        checks=checks,
        method="local_checks",
        safe_response=SAFE_VERIFICATION_RESPONSE,
    )


def _tool(
    name: str,
    description: str,
    properties: dict[str, tuple[str, str]],
    required: list[str] | None = None,
) -> dict[str, Any]:
    type_map = {
        "STRING": "string",
        "INTEGER": "integer",
    }
    encoded_properties = {}
    for key, (value_type, field_description) in properties.items():
        encoded_properties[key] = {
            "type": type_map.get(value_type, "string"),
            "description": field_description,
        }

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": encoded_properties,
                "required": required or [],
            },
        },
    }


def _verification_result(
    status: str,
    reason: str,
    checks: list[dict[str, Any]],
    method: str,
    safe_response: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "checks": checks,
        "method": method,
        "safe_response": safe_response,
    }


def _build_revision_feedback(query: str, answer: str, verification: dict[str, Any]) -> str:
    checks = verification.get("checks", [])
    check_lines = []
    for check in checks[:6]:
        if isinstance(check, dict):
            name = check.get("name", "check")
            status = check.get("status", "unknown")
            detail = check.get("detail", "")
            check_lines.append(f"- {name}: {status}. {detail}")
    if not check_lines:
        check_lines.append(f"- verification: {verification.get('reason', 'Unknown issue')}")

    return (
        "Logic verification failed for your previous answer.\n"
        "Revise the answer using ONLY the tool outputs already in this conversation.\n"
        "Do not invent new facts. If evidence is insufficient, say so clearly or ask one clarification question.\n\n"
        f"Original question: {query}\n"
        f"Previous answer: {answer}\n"
        f"Verification issue: {verification.get('reason', 'Unknown issue')}\n"
        "Checks:\n"
        + "\n".join(check_lines)
    )


def _recover_failed_tool_use(response: httpx.Response, step: int) -> dict[str, Any] | None:
    try:
        error_json = response.json()
        err = error_json.get("error", {})
        failed_generation = err.get("failed_generation", "")
        if err.get("code") != "tool_use_failed" or not failed_generation:
            return None

        match = re.search(r"<function=([a-zA-Z0-9_]+)(.*?)</function>", failed_generation, re.DOTALL)
        if not match:
            return None

        func_name = match.group(1)
        func_args_str = match.group(2).strip()
        if func_args_str.startswith(">"):
            func_args_str = func_args_str[1:].strip()

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"call_{step}_rev",
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": func_args_str,
                                },
                            }
                        ],
                    }
                }
            ]
        }
    except Exception:
        return None


def _collect_citations(
    call_name: str,
    result: Any,
    citations: list[dict[str, Any]],
    seen_filenames: set[str],
) -> None:
    if (
        call_name not in {"search_knowledge", "get_knowledge_context"}
        or not isinstance(result, dict)
        or "result" not in result
        or not isinstance(result["result"], list)
    ):
        return

    for item in result["result"]:
        if not isinstance(item, dict) or "document" not in item:
            continue
        filename = item.get("document", "Unknown")
        if filename in seen_filenames:
            continue
        seen_filenames.add(filename)
        citations.append(
            {
                "kind": "document",
                "asset_id": item.get("asset_id", ""),
                "original_filename": filename,
                "chunk_id": item.get("chunk_id", ""),
                "source_page": item.get("source_page"),
                "quote": _compact_text(str(item.get("content", "")), 200),
            }
        )


def _collect_dataset_sources(
    call_name: str,
    result: Any,
    dataset_sources: list[dict[str, Any]],
    seen_dataset_sources: set[str],
) -> None:
    if (
        call_name != "run_duckdb_sql"
        or not isinstance(result, dict)
        or not isinstance(result.get("dataset_sources"), list)
    ):
        return

    for item in result["dataset_sources"]:
        if not isinstance(item, dict):
            continue
        source_key = str(
            item.get("asset_id")
            or item.get("schema_name")
            or item.get("original_filename")
            or item.get("title")
            or ""
        )
        if not source_key or source_key in seen_dataset_sources:
            continue
        seen_dataset_sources.add(source_key)
        dataset_sources.append(
            {
                "kind": "dataset",
                "asset_id": item.get("asset_id", ""),
                "title": item.get("title", ""),
                "original_filename": item.get("original_filename", ""),
                "schema_name": item.get("schema_name", ""),
            }
        )


def _build_primary_sources(
    *,
    query: str,
    answer: str,
    tool_runs: list[ToolRun],
    citations: list[dict[str, Any]],
    dataset_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for run in reversed(tool_runs):
        if run.result != "success":
            continue
        if run.tool == "run_duckdb_sql":
            primary_dataset_sources = _dataset_sources_from_sql_run(run)
            if primary_dataset_sources:
                return primary_dataset_sources
        if run.tool in {"search_knowledge", "get_knowledge_context"}:
            primary_document_sources = _primary_document_sources_from_run(
                query=query,
                answer=answer,
                run=run,
            )
            if primary_document_sources:
                return primary_document_sources

    if dataset_sources:
        return [dict(source) for source in dataset_sources]
    if citations:
        return _primary_document_sources_from_citations(
            query=query,
            answer=answer,
            citations=citations,
        )
    return []


def _dataset_sources_from_sql_run(run: ToolRun) -> list[dict[str, Any]]:
    if not isinstance(run.output, dict):
        return []

    raw_sources = run.output.get("dataset_sources")
    if not isinstance(raw_sources, list):
        return []

    normalized_sources: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for item in raw_sources:
        if not isinstance(item, dict):
            continue
        key = str(
            item.get("asset_id")
            or item.get("schema_name")
            or item.get("original_filename")
            or item.get("title")
            or ""
        )
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        normalized_sources.append(
            {
                "kind": "dataset",
                "asset_id": item.get("asset_id", ""),
                "title": item.get("title", ""),
                "original_filename": item.get("original_filename", ""),
                "schema_name": item.get("schema_name", ""),
            }
        )

    return normalized_sources


def _primary_document_sources_from_run(
    *,
    query: str,
    answer: str,
    run: ToolRun,
) -> list[dict[str, Any]]:
    if not isinstance(run.output, dict):
        return []

    raw_items = run.output.get("result")
    if not isinstance(raw_items, list):
        return []

    return _select_best_document_sources(
        query=query,
        answer=answer,
        items=raw_items,
        filename_key="document",
        content_key="content",
    )


def _primary_document_sources_from_citations(
    *,
    query: str,
    answer: str,
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _select_best_document_sources(
        query=query,
        answer=answer,
        items=citations,
        filename_key="original_filename",
        content_key="quote",
    )


def _select_best_document_sources(
    *,
    query: str,
    answer: str,
    items: list[dict[str, Any]],
    filename_key: str,
    content_key: str,
) -> list[dict[str, Any]]:
    query_terms = _matching_terms(query)
    answer_terms = _matching_terms(answer)
    scored_documents: dict[str, dict[str, Any]] = {}

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        filename = str(item.get(filename_key, "") or "").strip()
        if not filename:
            continue

        source = {
            "kind": "document",
            "asset_id": item.get("asset_id", ""),
            "original_filename": filename,
            "chunk_id": item.get("chunk_id", ""),
            "source_page": item.get("source_page"),
            "quote": _compact_text(str(item.get(content_key, "")), 200),
        }
        haystack = " ".join(
            [
                filename,
                str(item.get("title", "")),
                str(item.get(content_key, "")),
            ]
        )
        score = _document_match_score(
            query_terms=query_terms,
            answer_terms=answer_terms,
            haystack=haystack,
        )
        key = str(item.get("asset_id") or filename)
        existing = scored_documents.get(key)
        if existing is None or score > existing["score"] or (
            score == existing["score"] and index < existing["index"]
        ):
            scored_documents[key] = {
                "score": score,
                "index": index,
                "source": source,
            }

    if not scored_documents:
        return []

    best = min(
        scored_documents.values(),
        key=lambda item: (
            -item["score"][0],
            -item["score"][1],
            item["index"],
        ),
    )
    return [dict(best["source"])]


def _document_match_score(
    *,
    query_terms: set[str],
    answer_terms: set[str],
    haystack: str,
) -> tuple[int, int]:
    normalized_haystack = _normalize_for_matching(haystack)
    answer_matches = sum(1 for term in answer_terms if term in normalized_haystack)
    query_matches = sum(1 for term in query_terms if term in normalized_haystack)
    return answer_matches, query_matches


def _matching_terms(value: str) -> set[str]:
    normalized = _normalize_for_matching(value)
    return {
        term
        for term in re.findall(r"[\w]+", normalized, flags=re.UNICODE)
        if len(term) >= 4
    }


def _normalize_for_matching(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold())
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return normalized.replace("\u0111", "d")


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _knowledge_evidence_is_empty(tool_runs: list[ToolRun]) -> bool:
    for run in tool_runs:
        if not isinstance(run.output, dict):
            return False
        result = run.output.get("result")
        if isinstance(result, list) and result:
            return False
        if isinstance(result, str) and "no relevant documents" not in result.casefold():
            return False
    return True


def _check_sql_alignment(answer: str, sql_runs: list[ToolRun]) -> dict[str, str] | None:
    if not sql_runs:
        return None

    latest_output = sql_runs[-1].output
    if not isinstance(latest_output, dict):
        return None

    answer_numbers = _extract_numbers(answer)
    candidate_numbers: set[str] = set()

    row_count = latest_output.get("total_rows", latest_output.get("row_count"))
    if isinstance(row_count, (int, float)):
        candidate_numbers.add(_normalize_number_token(row_count))

    rows = latest_output.get("data", latest_output.get("rows", []))
    if isinstance(rows, list):
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            for value in row.values():
                if isinstance(value, (int, float)):
                    candidate_numbers.add(_normalize_number_token(value))

    if row_count == 0 and not _looks_like_safe_answer(answer):
        return _check(
            "sql_alignment",
            "failed",
            "SQL returned zero rows, but the answer did not acknowledge missing data.",
        )

    if candidate_numbers and answer_numbers and candidate_numbers.isdisjoint(answer_numbers):
        return _check(
            "sql_alignment",
            "failed",
            "Numeric claims in the answer do not match the SQL output.",
        )

    if candidate_numbers:
        return _check(
            "sql_alignment",
            "passed",
            "Final answer stayed numerically consistent with SQL evidence.",
        )

    return _check(
        "sql_alignment",
        "not_applicable",
        "SQL output did not expose scalar values that require a numeric check.",
    )


def _extract_numbers(value: str) -> set[str]:
    return {_normalize_number_token(match) for match in re.findall(r"\d+(?:[.,]\d+)?", value)}


def _normalize_number_token(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return text
    return text.replace(",", "")


def _looks_like_safe_answer(answer: str) -> bool:
    normalized = answer.casefold()
    markers = [
        "khong tim thay",
        "khong co du lieu",
        "chua the xac minh",
        "can them",
        "clarification",
        "khong du thong tin",
        "no relevant",
        "not enough",
        "insufficient",
        "0",
    ]
    return any(marker in normalized for marker in markers)


def _build_semantic_verifier_prompt(
    query: str,
    answer: str,
    tool_runs: list[ToolRun],
    citations: list[dict[str, Any]],
    local_checks: list[dict[str, Any]],
) -> str:
    compact_tool_runs = [
        {
            "step": run.step,
            "tool": run.tool,
            "result": run.result,
            "args": run.args,
            "output": _compact_tool_output(run.output),
        }
        for run in tool_runs[-6:]
    ]
    compact_citations = [
        {
            "document": citation.get("original_filename"),
            "chunk_id": citation.get("chunk_id"),
            "quote": _compact_text(str(citation.get("quote", "")), 240),
            "source_page": citation.get("source_page"),
        }
        for citation in citations[:4]
    ]

    return (
        "Verify whether the candidate answer is fully supported by the available evidence.\n"
        "Rules:\n"
        "1. Mark `passed` only when the answer is grounded in the tool outputs.\n"
        "2. Mark `needs_revision` when the answer over-claims, mixes up numeric facts, or ignores a failed local check.\n"
        "3. Mark `insufficient_evidence` when the available evidence cannot support a safe final answer.\n"
        "4. Prefer a safe answer over guessing.\n\n"
        f"User question:\n{query}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        f"Local checks:\n{json.dumps(local_checks, ensure_ascii=False, default=str)}\n\n"
        f"Tool outputs:\n{json.dumps(compact_tool_runs, ensure_ascii=False, default=str)}\n\n"
        f"Citations:\n{json.dumps(compact_citations, ensure_ascii=False, default=str)}\n\n"
        "Return JSON only with this exact schema:\n"
        "{"
        '"status":"passed|needs_revision|insufficient_evidence",'
        '"reason":"short explanation",'
        '"checks":[{"name":"check_name","status":"passed|failed|not_applicable","detail":"short detail"}],'
        '"safe_response":"safe fallback answer for the user when evidence is weak"'
        "}"
    )


def _compact_tool_output(value: Any, max_chars: int = 700) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    return _compact_text(text, max_chars)


def _compact_text(value: str, max_chars: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."
