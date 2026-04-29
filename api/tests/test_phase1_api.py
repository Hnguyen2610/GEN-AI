from pathlib import Path
import shutil
from uuid import uuid4

from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.app.core.config import Settings, get_settings
from api.app.dependencies.auth import get_current_user
from api.app.dependencies.services import get_asset_service
from api.app.main import create_app
from api.app.repositories.in_memory import (
    InMemoryAssetRepository,
    InMemoryDatasetRepository,
    InMemoryKnowledgeRepository,
)
from api.app.repositories.interfaces import AuthenticatedUser
from api.app.services.assets import AssetService
from api.app.services.datasets import DatasetService
from api.app.services.knowledge import KnowledgeService
from api.app.services.parsers.tabular import TabularParser
from api.app.services.storage import LocalStorageService


test_user_context = {"user": None}


def build_test_client(max_upload_size_mb: float = 20):
    storage_root = Path("tmp/test-storage") / str(uuid4())
    storage_root.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        storage_backend="local",
        local_storage_root=str(storage_root),
        max_upload_size_mb=max_upload_size_mb,
        auto_create_schema=False,
    )
    app = create_app(settings)

    workspaces_by_user = {}
    dataset_repo = InMemoryDatasetRepository(workspaces_by_user)
    knowledge_repo = InMemoryKnowledgeRepository(workspaces_by_user)
    asset_repo = InMemoryAssetRepository(dataset_repo, knowledge_repo)
    storage_service = LocalStorageService(str(storage_root))
    tabular_parser = TabularParser(str(storage_root))
    dataset_service = DatasetService(
        repository=dataset_repo,
        storage_service=storage_service,
        settings=settings,
        tabular_parser=tabular_parser,
    )
    knowledge_service = KnowledgeService(
        repository=knowledge_repo,
        storage_service=storage_service,
        settings=settings,
    )
    asset_service = AssetService(asset_repo, dataset_service, knowledge_service, settings)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_asset_service] = lambda: asset_service

    def override_get_current_user():
        user = test_user_context["user"]
        if user is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return user

    app.dependency_overrides[get_current_user] = override_get_current_user
    return TestClient(app), storage_root


def test_auth_guard():
    client, storage_root = build_test_client()
    try:
        test_user_context["user"] = None
        response = client.get("/v1/assets")
        assert response.status_code == 401
    finally:
        shutil.rmtree(storage_root, ignore_errors=True)


def test_max_upload_size_limit():
    client, storage_root = build_test_client(max_upload_size_mb=0.001)
    try:
        test_user_context["user"] = AuthenticatedUser(
            user_id=uuid4(), email="u@e.com", access_token="t"
        )

        response = client.post(
            "/v1/assets/upload",
            files={"file": ("data.csv", b"x" * 2000, "text/csv")},
        )
        assert response.status_code == 413
    finally:
        shutil.rmtree(storage_root, ignore_errors=True)


def test_workspace_isolation():
    client, storage_root = build_test_client()
    try:
        user_a = AuthenticatedUser(user_id=uuid4(), email="a@e.com", access_token="t1")
        user_b = AuthenticatedUser(user_id=uuid4(), email="b@e.com", access_token="t2")

        test_user_context["user"] = user_a
        client.post("/v1/assets/upload", files={"file": ("a.csv", b"a", "text/csv")})
        client.post("/v1/assets/upload", files={"file": ("a.txt", b"a", "text/plain")})
        assert len(client.get("/v1/assets").json()["items"]) == 2

        test_user_context["user"] = user_b
        assert len(client.get("/v1/assets").json()["items"]) == 0
    finally:
        shutil.rmtree(storage_root, ignore_errors=True)


def test_unified_upload_routes_file_type_to_lane():
    client, storage_root = build_test_client()
    try:
        test_user_context["user"] = AuthenticatedUser(
            user_id=uuid4(), email="u@e.com", access_token="t"
        )

        dataset_response = client.post(
            "/v1/assets/upload",
            files={"file": ("data.csv", b"a,b\n1,2", "text/csv")},
        )
        assert dataset_response.status_code == 201
        assert dataset_response.json()["kind"] == "dataset"

        knowledge_response = client.post(
            "/v1/assets/upload",
            files={"file": ("doc.pdf", b"%PDF-1.4\nx", "application/pdf")},
        )
        assert knowledge_response.status_code == 201
        assert knowledge_response.json()["kind"] == "knowledge"

        unsupported_response = client.post(
            "/v1/assets/upload",
            files={"file": ("app.exe", b"x", "application/octet-stream")},
        )
        assert unsupported_response.status_code == 400
    finally:
        shutil.rmtree(storage_root, ignore_errors=True)


def test_real_wiring_probe():
    settings = Settings(auto_create_schema=False)
    app = create_app(settings)
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id=uuid4(), email="m@e.com", access_token="t"
    )
    app.dependency_overrides[get_settings] = lambda: settings

    client = TestClient(app)
    response = client.get("/v1/assets")
    assert response.status_code != 500 or "NameError" not in response.text
