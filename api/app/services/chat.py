import math
import uuid
from typing import Any, AsyncGenerator

from api.app.agent.pev import PEVAgentService
from api.app.agent.tools import AgentToolRegistry
from api.app.models.entities import MessageRole, MessageStatus
from api.app.repositories.chat import ChatRepository
from api.app.repositories.interfaces import AuthenticatedUser, DatasetRepository
from api.app.services.general_chat import GeneralChatService
from api.app.services.model_aliases import normalize_model_name
from api.app.services.rag import RAGService
from api.app.services.router import RouteResult, RouterService
from api.app.services.sql import TextToSQLService


CLARIFICATION_MESSAGE = (
    "Câu hỏi chưa đủ rõ ràng. Bạn muốn tôi tìm trong tài liệu, "
    "phân tích dữ liệu bảng, hay kết hợp cả hai?"
)


class ChatService:
    def __init__(
        self,
        repository: DatasetRepository,
        chat_repository: ChatRepository,
        rag_service: RAGService,
        sql_service: TextToSQLService,
        general_chat_service: GeneralChatService,
        router_service: RouterService | None = None,
        agent_service: PEVAgentService | None = None,
    ):
        self._repository = repository
        self._chat_repository = chat_repository
        self._rag_service = rag_service
        self._sql_service = sql_service
        self._general_chat_service = general_chat_service
        self._router_service = router_service
        self._agent_service = agent_service

    async def create_session(self, current_user: AuthenticatedUser, title: str):
        workspace = await self._repository.ensure_workspace_for_user(
            current_user.user_id, current_user.email
        )
        return await self._chat_repository.create_session(workspace.id, title)

    async def list_sessions(self, current_user: AuthenticatedUser, limit: int = 50):
        workspace = await self._repository.ensure_workspace_for_user(
            current_user.user_id, current_user.email
        )
        return await self._chat_repository.list_sessions(workspace.id, limit)

    async def get_session(self, current_user: AuthenticatedUser, session_id: uuid.UUID):
        workspace = await self._repository.ensure_workspace_for_user(
            current_user.user_id, current_user.email
        )
        return await self._chat_repository.get_session(workspace.id, session_id)

    async def delete_session(self, current_user: AuthenticatedUser, session_id: uuid.UUID) -> bool:
        workspace = await self._repository.ensure_workspace_for_user(
            current_user.user_id, current_user.email
        )
        return await self._chat_repository.delete_session(workspace.id, session_id)

    async def stream_message(
        self,
        current_user: AuthenticatedUser,
        session_id: uuid.UUID,
        content: str,
        model_choice: str,
    ) -> AsyncGenerator[tuple[str, dict | None, uuid.UUID], None]:
        resolved_model_choice = normalize_model_name(model_choice)
        workspace = await self._repository.ensure_workspace_for_user(
            current_user.user_id, current_user.email
        )
        chat_session = await self._chat_repository.get_session(workspace.id, session_id)
        if not chat_session:
            raise ValueError("Session not found in this workspace")

        route_result = await self._decide_route(workspace.id, content, resolved_model_choice)

        await self._chat_repository.create_message(
            session_id=session_id,
            role=MessageRole.user,
            content=content,
            status=MessageStatus.completed,
            commit=False,
        )
        assistant_msg = await self._chat_repository.create_message(
            session_id=session_id,
            role=MessageRole.assistant,
            content="",
            status=MessageStatus.streaming,
            model_name=resolved_model_choice,
            retrieval_used=route_result.route in {"rag", "hybrid", "agent"},
            commit=True,
        )

        history = [
            {"role": _enum_value(message.role), "content": message.content}
            for message in (chat_session.messages or [])
            if _enum_value(message.status) == MessageStatus.completed.value
        ]

        full_content = ""
        final_meta = None

        try:
            async for token, meta in self._stream_by_route(
                route_result=route_result,
                workspace_id=workspace.id,
                content=content,
                model_choice=resolved_model_choice,
                history=history,
            ):
                if token:
                    full_content += token
                    yield token, None, assistant_msg.id

                if meta:
                    final_meta = self._with_route_metadata(meta, route_result)
                    final_meta = _sanitize_metadata(final_meta)
                    yield "", final_meta, assistant_msg.id

            final_status = (
                MessageStatus.failed
                if final_meta and final_meta.get("error")
                else MessageStatus.completed
            )
            await self._chat_repository.update_message(
                message_id=assistant_msg.id,
                content=full_content,
                status=final_status,
                metadata_json=final_meta,
            )
        except Exception as exc:
            final_meta = self._with_route_metadata({"error": str(exc)}, route_result)
            final_meta = _sanitize_metadata(final_meta)
            await self._chat_repository.update_message(
                message_id=assistant_msg.id,
                content=full_content,
                status=MessageStatus.failed,
                metadata_json=final_meta,
            )
            raise

    async def _decide_route(
        self,
        workspace_id: uuid.UUID,
        content: str,
        model_choice: str,
    ) -> RouteResult:
        if self._router_service is None:
            return RouteResult(
                route="rag",
                reason="Router service is not configured.",
                confidence=0.5,
            )
        return await self._router_service.decide_route(workspace_id, content, model_choice)

    async def _stream_by_route(
        self,
        route_result: RouteResult,
        workspace_id: uuid.UUID,
        content: str,
        model_choice: str,
        history: list[dict],
    ) -> AsyncGenerator[tuple[str, dict | None], None]:
        if route_result.route == "chat":
            async for item in self._general_chat_service.stream_generation(
                model_choice, content, history
            ):
                yield item
            return

        if route_result.route == "rag":
            contexts = await self._rag_service.retrieve_context(workspace_id, content)
            async for item in self._rag_service.stream_generation(
                model_choice, content, contexts, history
            ):
                yield item
            return

        if route_result.route == "sql":
            async for item in self._sql_service.stream_generation(
                model_choice, content, workspace_id, history
            ):
                yield item
            return

        if route_result.route in {"hybrid", "agent"}:
            if self._agent_service is None:
                yield "Agent chưa được cấu hình để xử lý câu hỏi kết hợp.", {
                    "route": route_result.route,
                    "error": "Agent service is not configured",
                }
                return

            registry = AgentToolRegistry(
                workspace_id=workspace_id,
                rag_service=self._rag_service,
                sql_service=self._sql_service,
            )
            async for item in self._agent_service.stream_response(
                workspace_id=workspace_id,
                query=content,
                history=history,
                model_name=model_choice,
                registry=registry,
            ):
                yield item
            return

        yield CLARIFICATION_MESSAGE, None
        yield "", {"route": "clarification", "error": None}

    @staticmethod
    def _with_route_metadata(meta: dict | None, route_result: RouteResult) -> dict:
        data = dict(meta or {})
        data.setdefault("route", route_result.route)
        data["route_reason"] = route_result.reason
        data["route_confidence"] = route_result.confidence
        return data


def _enum_value(value):
    return getattr(value, "value", value)


def _sanitize_metadata(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _sanitize_metadata(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_metadata(i) for i in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
    return data
