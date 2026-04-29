from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile, status

from api.app.core.config import Settings
from api.app.repositories.interfaces import (
    AuthenticatedUser,
    CreateDatasetBundlePayload,
    DatasetRecord,
    DatasetRepository,
    JobRecord,
)
from api.app.schemas.datasets import (
    DatasetResponse, 
    DatasetVersionSummary, 
    JobSummary,
    DatasetPreviewResponse,
    DatasetProfileResponse,
    DatasetSheetSummary,
    ColumnProfileSummary
)
from api.app.schemas.jobs import JobResponse
from api.app.services.storage import StorageService
from api.app.services.parsers.tabular import TabularParser


class DatasetService:
    def __init__(self, *, repository: DatasetRepository, storage_service: StorageService, settings: Settings, tabular_parser: TabularParser):
        self._repository = repository
        self._storage_service = storage_service
        self._settings = settings
        self._tabular_parser = tabular_parser

    async def create_pending_dataset(
        self, current_user: AuthenticatedUser, upload_file: UploadFile
    ) -> tuple[DatasetResponse, JobSummary]:
        workspace = await self._repository.ensure_workspace_for_user(current_user.user_id, current_user.email)
        original_filename = Path((upload_file.filename or "").strip()).name

        if not original_filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required.")

        extension = Path(original_filename).suffix.lower().lstrip(".")
        if extension not in self._settings.allowed_dataset_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported dataset file type '.{extension or 'unknown'}'.",
            )

        content = await upload_file.read()
        if not content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

        if len(content) > self._settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds {self._settings.max_upload_size_mb} MB limit.",
            )

        dataset_id = uuid4()
        dataset_version_id = uuid4()
        job_id = uuid4()
        storage_path = f"{workspace.id}/{dataset_id}/v1/{original_filename}"
        content_type = upload_file.content_type or _guess_content_type(extension)

        stored_file = await self._storage_service.save(
            path=storage_path,
            content=content,
            content_type=content_type,
        )

        payload = CreateDatasetBundlePayload(
            workspace_id=workspace.id,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            job_id=job_id,
            created_by=current_user.user_id,
            title=Path(original_filename).stem,
            original_filename=original_filename,
            mime_type=content_type,
            storage_backend=stored_file.backend,
            storage_path=stored_file.path,
            file_size_bytes=len(content),
            checksum_sha256=sha256(content).hexdigest(),
        )

        try:
            dataset_record, job_record = await self._repository.create_dataset_bundle(payload)
        except Exception:
            await self._storage_service.delete(path=stored_file.path)
            raise

        return self._map_dataset(dataset_record), self._map_job(job_record)

    async def list_datasets(self, current_user: AuthenticatedUser) -> list[DatasetResponse]:
        workspace = await self._repository.ensure_workspace_for_user(current_user.user_id, current_user.email)
        datasets = await self._repository.list_datasets(workspace.id)
        return [self._map_dataset(dataset) for dataset in datasets]

    async def get_job(self, current_user: AuthenticatedUser, job_id: UUID) -> JobSummary:
        workspace = await self._repository.ensure_workspace_for_user(current_user.user_id, current_user.email)
        job = await self._repository.get_job(workspace.id, job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        return JobResponse(**self._map_job(job).model_dump())


    async def get_preview(
        self, current_user: AuthenticatedUser, dataset_id: UUID, sheet_name: str | None = None
    ) -> DatasetPreviewResponse:
        workspace = await self._repository.ensure_workspace_for_user(current_user.user_id, current_user.email)
        # Fetch latest version to verify access and get metadata if needed
        datasets = await self._repository.list_datasets(workspace.id)
        dataset = next((d for d in datasets if d.id == dataset_id), None)
        
        if not dataset:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied.")
            
        preview_data = self._tabular_parser.get_preview(workspace.id, dataset_id, sheet_name)
        return DatasetPreviewResponse(**preview_data)

    async def get_profile(
        self, current_user: AuthenticatedUser, dataset_id: UUID
    ) -> DatasetProfileResponse:
        workspace = await self._repository.ensure_workspace_for_user(current_user.user_id, current_user.email)
        datasets = await self._repository.list_datasets(workspace.id)
        dataset = next((d for d in datasets if d.id == dataset_id), None)
        
        if not dataset or not dataset.latest_version:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset version not found.")

        sheets = await self._repository.get_dataset_sheets(dataset.latest_version.id)
        profiles = await self._repository.get_column_profiles(dataset.latest_version.id)
        
        return DatasetProfileResponse(
            sheets=[
                DatasetSheetSummary(
                    id=s.id, name=s.name, row_count=s.row_count, column_count=s.column_count, created_at=s.created_at
                ) for s in sheets
            ],
            columns=[
                ColumnProfileSummary(
                    id=p.id, sheet_name=p.sheet_name, column_name=p.column_name, data_type=p.data_type,
                    null_count=p.null_count, distinct_count=p.distinct_count, min_value=p.min_value,
                    max_value=p.max_value, sample_values=p.sample_values
                ) for p in profiles
            ]
        )

    async def delete_dataset(self, current_user: AuthenticatedUser, dataset_id: UUID) -> None:
        workspace = await self._repository.ensure_workspace_for_user(current_user.user_id, current_user.email)
        
        # Get info before deletion to cleanup storage
        datasets = await self._repository.list_datasets(workspace.id)
        dataset = next((d for d in datasets if d.id == dataset_id), None)
        
        if not dataset:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

        # 1. Cleanup Storage (V1)
        if dataset.latest_version and dataset.latest_version.storage_path:
            await self._storage_service.delete(path=dataset.latest_version.storage_path)

        # 2. Cleanup DuckDB Materialization
        self._tabular_parser.delete_materialization(workspace.id, dataset_id)

        # 3. Delete from Repository (DB records - cascading)
        await self._repository.delete_dataset(workspace.id, dataset_id)

    @staticmethod
    def _map_dataset(dataset: DatasetRecord) -> DatasetResponse:
        latest_version = None
        if dataset.latest_version is not None:
            latest_version = DatasetVersionSummary(
                id=dataset.latest_version.id,
                version_number=dataset.latest_version.version_number,
                file_size_bytes=dataset.latest_version.file_size_bytes,
                created_at=dataset.latest_version.created_at,
            )

        latest_job = None
        if dataset.latest_job is not None:
            latest_job = JobSummary(
                id=dataset.latest_job.id,
                status=dataset.latest_job.status,
                created_at=dataset.latest_job.created_at,
                updated_at=dataset.latest_job.updated_at,
                error_message=dataset.latest_job.error_message,
            )

        return DatasetResponse(
            id=dataset.id,
            workspace_id=dataset.workspace_id,
            title=dataset.title,
            original_filename=dataset.original_filename,
            mime_type=dataset.mime_type,
            status=dataset.status,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            latest_version=latest_version,
            latest_job=latest_job,
        )

    @staticmethod
    def _map_job(job: JobRecord) -> JobSummary:
        return JobSummary(
            id=job.id,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            error_message=job.error_message,
        )


def _guess_content_type(extension: str) -> str:
    return {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain",
        "md": "text/markdown",
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel",
    }.get(extension, "application/octet-stream")
