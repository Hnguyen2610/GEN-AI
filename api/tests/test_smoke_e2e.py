import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from fastapi.testclient import TestClient
from api.app.dependencies.services import get_chat_service
from api.app.main import app
from api.app.dependencies.auth import get_current_user
from api.app.repositories.interfaces import AuthenticatedUser
from api.app.models.entities import Workspace, Asset, AssetKind, DatasetStatus
from api.app.schemas.assets import UploadAssetResponse

# Override Auth
def override_get_current_user():
    return AuthenticatedUser(user_id=uuid4(), email="tester@example.com", access_token="mock_token")

app.dependency_overrides[get_current_user] = override_get_current_user

# We will test the routes from top-level to ensure imports/injections work
client = TestClient(app)

@pytest.fixture
def mock_db_session():
    session = AsyncMock()
    return session

@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.upload.return_value = "mock_path.csv"
    return storage

@pytest.mark.anyio
@patch('api.app.dependencies.services.get_db_session')
@patch('api.app.dependencies.services.build_storage_service')
@patch('api.app.services.assets.AssetService.upload_asset', new_callable=AsyncMock)
def test_smoke_e2e_asset_upload(mock_upload, mock_build_storage, mock_get_db):
    """
    Test that the deprecated routes are gone and the new unified route handles uploads.
    """
    # Verify legacy datasets router is gone
    res_legacy = client.post("/v1/datasets/upload")
    assert res_legacy.status_code == 404
    
    # Mock behavior for unified asset
    mock_upload.return_value = UploadAssetResponse(
        asset_id=uuid4(), 
        kind=AssetKind.dataset, 
        job_id=uuid4(),
        status=DatasetStatus.processing
    )
    
    # Hit unified endpoint
    with open(__file__, "rb") as f:
         response = client.post(
             "/v1/assets/upload",
             data={"kind": "dataset", "title": "Test CSV", "description": "Smoke context"},
             files={"file": ("test.csv", f, "text/csv")}
         )
         
    # Assuming standard success returns 200 or 201 with data
    assert response.status_code in [200, 201]
    
@pytest.mark.anyio
def test_smoke_e2e_chat_agent():
    """
    Test that chat endpoints wire up properly to the streaming API surface.
    """
    workspace_id = uuid4()
    session_id = uuid4()

    class FakeChatService:
        async def create_session(self, user, title):
            return {
                "id": str(session_id),
                "workspace_id": str(workspace_id),
                "title": title,
            }

        async def stream_message(self, current_user, session_id, content, model_choice):
            yield "Hello from agent", None, uuid4()
            yield "", {
                "route": "agent",
                "agent_traces": [],
                "verification": {"status": "passed"},
            }, uuid4()

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    try:
        session_response = client.post("/v1/chat/sessions", json={"title": "Smoke"})
        assert session_response.status_code == 201
        assert session_response.json()["id"] == str(session_id)

        message_response = client.post(
            f"/v1/chat/sessions/{session_id}/messages",
            json={"content": "hello", "model_choice": "gemini-2.5-flash"},
        )
        assert message_response.status_code == 200
        assert "event: token" in message_response.text
        assert "event: end" in message_response.text
        assert "Hello from agent" in message_response.text
    finally:
        app.dependency_overrides.pop(get_chat_service, None)
