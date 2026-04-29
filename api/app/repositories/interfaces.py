
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from api.app.models.entities import DatasetSheet, ColumnProfile, KnowledgeChunk, AssetKind


@dataclass(slots=True)
class AuthenticatedUser:
    user_id: UUID
    email: str | None
    access_token: str


@dataclass(slots=True)
class WorkspaceRecord:
    id: UUID
    slug: str
    name: str


@dataclass(slots=True)
class AssetVersionRecord:
    id: UUID
    version_number: int
    storage_path: str
    file_size_bytes: int
    created_at: datetime


@dataclass(slots=True)
class DatasetVersionRecord:
    id: UUID
    version_number: int
    storage_path: str
    file_size_bytes: int
    created_at: datetime


@dataclass(slots=True)
class JobRecord:
    id: UUID
    status: str
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None


@dataclass(slots=True)
class DatasetRecord:
    id: UUID
    workspace_id: UUID
    title: str
    original_filename: str
    mime_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    latest_version: DatasetVersionRecord | None
    latest_job: JobRecord | None


@dataclass(slots=True)
class CreateDatasetBundlePayload:
    workspace_id: UUID
    dataset_id: UUID
    dataset_version_id: UUID
    job_id: UUID
    created_by: UUID
    title: str
    original_filename: str
    mime_type: str
    storage_backend: str
    storage_path: str
    file_size_bytes: int
    checksum_sha256: str


@dataclass(slots=True)
class KnowledgeVersionRecord:
    id: UUID
    version_number: int
    storage_path: str
    file_size_bytes: int
    created_at: datetime


@dataclass(slots=True)
class KnowledgeRecord:
    id: UUID
    workspace_id: UUID
    title: str
    original_filename: str
    mime_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    latest_version: KnowledgeVersionRecord | None
    latest_job: JobRecord | None


@dataclass(slots=True)
class CreateKnowledgeAssetPayload:
    workspace_id: UUID
    knowledge_asset_id: UUID
    knowledge_version_id: UUID
    job_id: UUID
    created_by: UUID
    title: str
    original_filename: str
    mime_type: str
    storage_backend: str
    storage_path: str
    file_size_bytes: int
    checksum_sha256: str


@dataclass(slots=True)
class DatasetSheetRecord:
    id: UUID
    dataset_id: UUID
    dataset_version_id: UUID
    name: str
    row_count: int | None
    column_count: int | None
    created_at: datetime
    asset_version_id: UUID | None = None


@dataclass(slots=True)
class ColumnProfileRecord:
    id: UUID
    dataset_id: UUID
    dataset_version_id: UUID
    sheet_name: str | None
    column_name: str
    data_type: str
    null_count: int | None
    distinct_count: int | None
    min_value: str | None
    max_value: str | None
    sample_values: dict | None
    created_at: datetime
    asset_version_id: UUID | None = None


@dataclass(slots=True)
class KnowledgeChunkRecord:
    id: UUID
    knowledge_version_id: UUID
    content: str
    embedding: list[float] | None
    metadata_json: dict
    chunk_index: int
    created_at: datetime
    asset_version_id: UUID | None = None


@dataclass(slots=True)
class AssetSummaryRecord:
    id: UUID
    kind: AssetKind
    title: str
    original_filename: str
    status: str
    created_at: datetime
    updated_at: datetime
    latest_job: JobRecord | None


@dataclass(slots=True)
class AssetDetailRecord:
    id: UUID
    kind: AssetKind
    title: str
    original_filename: str
    mime_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    latest_version: AssetVersionRecord | None
    latest_job: JobRecord | None


class AssetRepository(Protocol):
    async def ensure_workspace_for_user(self, user_id: UUID, email: str | None) -> WorkspaceRecord: ...
    async def list_assets(self, workspace_id: UUID) -> list[AssetSummaryRecord]: ...
    async def get_asset(self, workspace_id: UUID, asset_id: UUID) -> AssetDetailRecord | None: ...
    async def delete_asset(self, workspace_id: UUID, asset_id: UUID) -> None: ...
    async def get_asset_preview(self, workspace_id: UUID, asset_id: UUID) -> list[dict] | None: ...
    async def get_asset_profile(self, workspace_id: UUID, asset_id: UUID) -> dict | None: ...


@dataclass(slots=True)
class ChatMessageRecord:
    id: UUID
    session_id: UUID
    role: str
    content: str
    status: str
    model_name: str | None
    retrieval_used: bool
    metadata_json: dict | None
    created_at: datetime


@dataclass(slots=True)
class ChatSessionRecord:
    id: UUID
    workspace_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageRecord] | None = None


from api.app.models.entities import MessageRole, MessageStatus

class ChatRepository(Protocol):
    async def create_session(self, workspace_id: UUID, title: str) -> ChatSessionRecord: ...
    async def get_session(self, workspace_id: UUID, session_id: UUID) -> ChatSessionRecord | None: ...
    async def list_sessions(self, workspace_id: UUID, limit: int = 50) -> list[ChatSessionRecord]: ...
    async def delete_session(self, workspace_id: UUID, session_id: UUID) -> bool: ...
    async def create_message(
        self, 
        session_id: UUID, 
        role: MessageRole, 
        content: str, 
        status: MessageStatus,
        model_name: str | None = None,
        retrieval_used: bool = False,
        metadata_json: dict | None = None,
        commit: bool = True
    ) -> ChatMessageRecord: ...
    async def update_message(
        self,
        message_id: UUID,
        content: str,
        status: MessageStatus,
        metadata_json: dict | None = None
    ) -> ChatMessageRecord | None: ...


class DatasetRepository(Protocol):
    async def ensure_workspace_for_user(self, user_id: UUID, email: str | None) -> WorkspaceRecord: ...

    async def create_dataset_bundle(
        self, payload: CreateDatasetBundlePayload
    ) -> tuple[DatasetRecord, JobRecord]: ...

    async def save_dataset_metadata(
        self, 
        sheets: list[DatasetSheetRecord], 
        profiles: list[ColumnProfileRecord]
    ) -> None: ...

    async def list_datasets(self, workspace_id: UUID) -> list[DatasetRecord]: ...

    async def get_job(self, workspace_id: UUID, job_id: UUID) -> JobRecord | None: ...

    async def get_dataset_sheets(self, dataset_version_id: UUID) -> list[DatasetSheetRecord]: ...

    async def get_column_profiles(self, dataset_version_id: UUID) -> list[ColumnProfileRecord]: ...

    async def delete_dataset(self, workspace_id: UUID, dataset_id: UUID) -> None: ...


class KnowledgeRepository(Protocol):
    async def ensure_workspace_for_user(self, user_id: UUID, email: str | None) -> WorkspaceRecord: ...

    async def create_knowledge_asset(
        self, payload: CreateKnowledgeAssetPayload
    ) -> tuple[KnowledgeRecord, JobRecord]: ...

    async def save_knowledge_chunks(
        self, 
        chunks: list[KnowledgeChunkRecord]
    ) -> None: ...

    async def list_knowledge(self, workspace_id: UUID) -> list[KnowledgeRecord]: ...

    async def delete_knowledge_asset(self, workspace_id: UUID, knowledge_id: UUID) -> None: ...
