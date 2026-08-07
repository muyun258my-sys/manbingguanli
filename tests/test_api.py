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


def test_health_returns_status_and_disclaimer():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 0
    assert payload["data"]["status"] == "healthy"
    assert payload["data"]["dependencies"]["profile_store"] is True
    assert payload["disclaimer"]


def test_chat_normal_request_returns_envelope():
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"session_id": "api-1", "user_id": "api-user-1", "message": "最近头晕，血压 160/100"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 0
    assert payload["data"]["intent"] == "symptom_analysis"
    assert payload["data"]["emergency"] is False
    assert payload["data"]["reply"]
    assert payload["data"]["sources"]
    assert payload["disclaimer"]


def test_chat_emergency_short_circuits():
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"session_id": "api-2", "user_id": "api-user-2", "message": "突然胸口剧痛，左臂也麻了"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["emergency"] is True
    assert payload["data"]["intent"] == "high_risk_input"
    assert "急救" in payload["data"]["reply"] or "急诊" in payload["data"]["reply"]


def test_chat_rejects_missing_fields():
    client = TestClient(app)

    response = client.post("/chat", json={"session_id": "api-3", "user_id": "api-user-3"})

    assert response.status_code == 422


def test_profile_roundtrip_via_api():
    client = TestClient(app)

    put = client.put(
        "/profile/api-user-4",
        json={
            "condition_description": "反复头晕两周",
            "conditions": ["高血压"],
            "medications": ["氨氯地平"],
            "allergies": ["青霉素"],
        },
    )
    assert put.status_code == 200
    assert put.json()["data"]["conditions"] == ["高血压"]

    get = client.get("/profile/api-user-4")
    assert get.status_code == 200
    profile = get.json()["data"]
    assert profile["condition_description"] == "反复头晕两周"
    assert profile["medications"] == ["氨氯地平"]
    assert profile["allergies"] == ["青霉素"]


def test_profile_partial_update_via_api():
    client = TestClient(app)

    client.put("/profile/api-user-5", json={"conditions": ["糖尿病"]})
    client.put("/profile/api-user-5", json={"medications": ["二甲双胍"]})

    profile = client.get("/profile/api-user-5").json()["data"]
    assert profile["conditions"] == ["糖尿病"]
    assert profile["medications"] == ["二甲双胍"]


def test_profile_unknown_user_returns_empty_profile():
    client = TestClient(app)

    profile = client.get("/profile/definitely-not-registered").json()["data"]

    assert profile["conditions"] == []
    assert profile["medications"] == []
    assert profile["allergies"] == []
