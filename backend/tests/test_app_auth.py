import pytest
from fastapi import HTTPException
import app as effectlab_app


def test_local_mode_remains_frictionless(monkeypatch):
    monkeypatch.setattr(effectlab_app, "CLOUD_MODE", False)
    monkeypatch.setattr(effectlab_app, "ACCESS_TOKEN", "")
    effectlab_app._require_auth(None)


def test_cloud_mode_requires_a_configured_token(monkeypatch):
    monkeypatch.setattr(effectlab_app, "CLOUD_MODE", True)
    monkeypatch.setattr(effectlab_app, "ACCESS_TOKEN", "")
    with pytest.raises(HTTPException) as exc:
        effectlab_app._require_auth(None)
    assert exc.value.status_code == 503


def test_cloud_mode_rejects_missing_or_wrong_token(monkeypatch):
    monkeypatch.setattr(effectlab_app, "CLOUD_MODE", True)
    monkeypatch.setattr(effectlab_app, "ACCESS_TOKEN", "unit-test-secret")
    with pytest.raises(HTTPException) as missing:
        effectlab_app._require_auth(None)
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException) as wrong:
        effectlab_app._require_auth("Bearer wrong")
    assert wrong.value.status_code == 401
    effectlab_app._require_auth("Bearer unit-test-secret")


def test_cloud_mode_disables_training_endpoints(monkeypatch):
    monkeypatch.setattr(effectlab_app, "CLOUD_MODE", True)
    monkeypatch.setattr(effectlab_app, "ACCESS_TOKEN", "unit-test-secret")
    with pytest.raises(HTTPException) as exc:
        effectlab_app._require_auth("Bearer unit-test-secret", training=True)
    assert exc.value.status_code == 403


def test_cloud_middleware_guards_uploads_and_allows_preflight(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(effectlab_app, "CLOUD_MODE", True)
    monkeypatch.setattr(effectlab_app, "ACCESS_TOKEN", "unit-test-secret")
    allowed_origin = "https://willoughbyorchard-cpu.github.io"
    with TestClient(effectlab_app.app) as client:
        # The auth middleware should answer before FastAPI asks for upload fields.
        denied = client.post("/api/jobs", headers={"Origin": allowed_origin}, json={})
        assert denied.status_code == 401
        assert "access token" in denied.json()["detail"].lower()

        # Browser CORS preflight must not be blocked by token enforcement.
        preflight = client.options(
            "/api/jobs",
            headers={
                "Origin": allowed_origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers.get("access-control-allow-origin") == allowed_origin

        disabled = client.get(
            "/api/training/dataset",
            headers={"Origin": allowed_origin, "Authorization": "Bearer unit-test-secret"},
        )
        assert disabled.status_code == 403
