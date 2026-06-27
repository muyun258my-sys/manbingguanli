from fastapi.testclient import TestClient

from app.app import app


def test_root_route_describes_backend():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 0
    assert payload["message"] == "backend is running"
    assert payload["data"]["health"] == "/health"
