"""The HTTP interface exercises a real SQLite queue and all forecast agents."""
from datetime import date, timedelta
from time import monotonic, sleep

from fastapi.testclient import TestClient
import pytest

from backend.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test:
        yield test


def user(client, name: str) -> dict:
    auth = {"email": f"{name}@example.org", "password": "test-password-123"}
    registered = client.post("/auth/register", json=auth)
    assert registered.status_code == 201, registered.text
    login = client.post("/auth/login", json=auth)
    assert login.status_code == 200
    assert client.post("/auth/register", json=auth).status_code == 409
    assert client.post("/auth/login", json={**auth, "password": "wrong-password"}).status_code == 401
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    assert client.get("/auth/me", headers=headers).json()["role"] == "user"
    return headers


def wait(client, endpoint: str, headers: dict, desired: set[str]) -> dict:
    end = monotonic() + 20
    while monotonic() < end:
        response = client.get(endpoint, headers=headers)
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] in desired:
            return payload
        if payload["status"] == "failed":
            pytest.fail(str(payload))
        sleep(.05)
    pytest.fail(f"Timed out waiting for {endpoint}")


def test_upload_forecast_trace_and_download(client):
    assert client.get("/").status_code == 200
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/api/v1/forecast/").status_code == 401
    headers = user(client, "series")
    data = "date;value\n" + "\n".join(
        f"{date(2024, 1, 1) + timedelta(days=i)};{100 + i / 2 + i % 7}" for i in range(70)
    )
    uploaded = client.post("/api/v1/forecast/", headers=headers, data={"horizon": 5, "mode": "+agents"},
                           files={"file": ("series.csv", data, "text/csv")})
    assert uploaded.status_code == 202, uploaded.text
    task_id = uploaded.json()["task_id"]
    path = f"/api/v1/forecast/{task_id}"
    wait(client, path + "/status", headers, {"completed", "waiting_for_human"})
    result = client.get(path + "/result", headers=headers).json()
    assert len(result["predictions"]) == 5
    assert result["interval"]["level"] == .8
    assert result["metrics"]["mae"] >= 0
    assert client.get(result["download_url"], headers=headers).text.count("\n") == 6
    tree = client.get(path + "/reasoning", headers=headers).json()["spans"]
    assert [item["agent"] for item in tree] == ["Supervisor"]
    iteration = next(item for item in tree[0]["children"] if item["agent"] == "Iteration")
    assert {item["agent"] for item in iteration["children"]} == {"ModelSelector", "Forecaster", "Validator", "Critic"}
    selector = iteration["children"][0]
    assert selector["llm_calls"][0]["tool_calling"]
    assert selector["children"][0]["agent"] == "HybridRetriever"
    outsider = user(client, "outsider")
    assert client.get(path + "/result", headers=outsider).status_code == 404


def test_housing_human_review_and_retry(client):
    headers = user(client, "apartment")
    payload = {"data": {"task_type": "tabular", "apartment": {
        "district": "Нет аналогов", "area_total": 52, "floor": 6, "floors_total": 9, "build_year": 1980,
    }}, "horizon": 3}
    response = client.post("/api/v1/forecast/", headers=headers, json=payload)
    assert response.status_code == 202, response.text
    path = "/api/v1/forecast/" + response.json()["task_id"]
    wait(client, path + "/status", headers, {"waiting_for_human"})
    result = client.get(path + "/result", headers=headers).json()
    assert result["data_origin"] == "synthetic_kazan_demo"
    assert result["trajectory"]["iterations"] == 3
    assert result["predicted_price"] > 0
    assert len(result["factors"]) == 5
    queue = client.get("/api/v1/forecast/?review_queue=true", headers=headers).json()
    assert len(queue) == 1
    assert client.post(path + "/hitl", headers=headers, json={"action": "retry", "horizon": 13}).status_code == 422
    resolution = {"action": "retry", "horizon": 2, "model": "fit_ensemble"}
    assert client.post(path + "/hitl", headers=headers, json=resolution).json()["status"] == "pending"
    current = wait(client, path + "/status", headers, {"waiting_for_human"})
    assert current["horizon"] == 2
    assert current["review"]["model"] == "fit_ensemble"
    retry_result = client.get(path + "/result", headers=headers).json()
    assert retry_result["model"]["name"] == "fit_ensemble"
    assert retry_result["trajectory"]["iterations"] == 1
    accepted = client.post(path + "/hitl", headers=headers, json={"action": "accept", "comment": "Проверено"})
    assert accepted.json()["status"] == "completed"
    assert accepted.json()["review"]["knowledge_case_id"].startswith("review:")
    result = client.get(path + "/result", headers=headers).json()
    assert result["accepted_by_human"] and not result["needs_review"]
    assert len(result["predictions"]) == 2
    assert client.post(path + "/hitl", headers=headers, json={"action": "reject"}).status_code == 409


def test_batch_evaluation_and_invalid_input(client):
    headers = user(client, "experiments")
    assert client.get("/api/v1/forecast/demo/series").json()["frequency"] == "QS"
    invalid = client.post("/api/v1/forecast/", json={"data": {}}, headers=headers)
    assert invalid.status_code == 422
    assert client.post("/api/v1/evaluate/batch", headers=headers, json={"horizons": [51]}).status_code == 422
    assert client.post("/api/v1/evaluate/batch", headers=headers, json={"configured_llm": True}).status_code == 403
    response = client.post("/api/v1/evaluate/batch", headers=headers,
                           json={"dataset": "tatarstan", "num_series": 1, "horizons": [2],
                                 "data_fractions": [.2, 1]})
    assert response.status_code == 202, response.text
    query = "/api/v1/evaluate/compare?evaluation_id=" + response.json()["evaluation_id"]
    result = wait(client, query, headers, {"completed"})
    assert len(result["summary"]) == 3
    assert all(row["fraction"] == 1 for row in result["summary"])
    assert len(result["comparisons"]) == 3


def test_baseline_reuses_own_result(client):
    headers = user(client, "cache")
    dataset = client.get("/api/v1/forecast/demo/series").json()
    payload = {"data": dataset, "horizon": 2, "mode": "baseline"}
    paths = []
    for _ in range(2):
        queued = client.post("/api/v1/forecast/", headers=headers, json=payload)
        assert queued.status_code == 202
        path = "/api/v1/forecast/" + queued.json()["task_id"]
        wait(client, path + "/status", headers, {"completed"})
        paths.append(path)
    prior, repeated = [client.get(path + "/result", headers=headers).json() for path in paths]
    assert prior["predictions"] == repeated["predictions"]
    assert repeated["trajectory"]["cache_hit"]
    assert client.get(paths[1] + "/reasoning", headers=headers).json()["spans"][0]["agent"] == "ForecastCache"


def test_settings_are_private_and_used_by_worker(client, monkeypatch):
    import json
    import sqlite3
    from pathlib import Path
    import httpx

    from backend.core.config import get_settings

    calls = []
    original = httpx.AsyncClient

    def respond(request):
        content = json.loads(request.content)
        if request.url.host == "api.deepseek.com":
            assert request.headers["Authorization"] == "Bearer integration-token"
            assert content["model"] == "integration-model"
        else:
            assert request.url.host == "openrouter.ai"
            assert "Authorization" not in request.headers
            assert content["model"] == "new-model"
        calls.append(bool(content.get("tools")))
        if content.get("tools"):
            function = content["tools"][0]["function"]["name"]
            message = {"tool_calls": [{"function": {"name": function, "arguments": "{}"}}]}
        else:
            message = {"content": '{"score":0.95,"comment":"Подходит для этих данных."}'}
        return httpx.Response(200, json={"choices": [{"message": message}],
                                         "usage": {"prompt_tokens": 14, "completion_tokens": 5}})

    monkeypatch.setattr("backend.core.llm.manager.httpx.AsyncClient",
                        lambda **kw: original(transport=httpx.MockTransport(respond),
                                              **{k: v for k, v in kw.items() if k != "transport"}))

    owner = user(client, "llm-owner")
    guest = user(client, "llm-guest")
    base = "/api/v1/llm"
    assert client.get(base + "/settings").status_code == 401
    bad = client.put(base + "/settings", headers=owner, json={
        "provider": "compatible", "model": "test", "endpoint": "https://unlisted.invalid/v1",
    })
    assert bad.status_code == 422
    response = client.put(base + "/settings", headers=owner, json={
        "provider": "deepseek", "model": "integration-model", "api_key": "integration-token",
    })
    assert response.status_code == 200, response.text
    assert response.json()["profile"]["has_api_key"]
    assert "integration-token" not in response.text
    database = Path(get_settings().DATABASE_URL.removeprefix("sqlite+aiosqlite:///"))
    with sqlite3.connect(database) as connection:
        ciphertext = connection.execute("SELECT encrypted_key FROM llm_profiles").fetchone()[0]
    assert ciphertext and "integration-token" not in ciphertext
    assert client.get(base + "/settings", headers=guest).json()["profile"] is None
    assert client.post(base + "/probe", headers=owner).json()["output_tokens"] == 5

    # Submission has no credential. The worker uses the task owner's profile.
    data = client.get("/api/v1/forecast/demo/series").json()
    queued = client.post("/api/v1/forecast/", headers=owner, json={"data": data, "horizon": 2})
    path = "/api/v1/forecast/" + queued.json()["task_id"]
    wait(client, path + "/status", owner, {"completed", "waiting_for_human"})
    result = client.get(path + "/result", headers=owner).json()
    assert result["trajectory"]["tokens"] > 0
    assert True in calls and False in calls

    changed = client.put(base + "/settings", headers=owner, json={
        "provider": "compatible", "model": "new-model", "endpoint": "https://openrouter.ai/api/v1",
    })
    assert changed.status_code == 200
    assert not changed.json()["profile"]["has_api_key"]
    assert client.delete(base + "/settings", headers=owner).status_code == 204
    assert client.get(base + "/settings", headers=owner).json()["profile"] is None


def test_profile_save_surfaces_provider_rejection(client, monkeypatch):
    import httpx

    original = httpx.AsyncClient
    monkeypatch.setattr("backend.core.llm.manager.httpx.AsyncClient", lambda **kw: original(
        transport=httpx.MockTransport(lambda request: httpx.Response(401)),
        **{k: v for k, v in kw.items() if k != "transport"},
    ))
    owner = user(client, "llm-rejected")
    response = client.put("/api/v1/llm/settings", headers=owner,
                          json={"provider": "deepseek", "api_key": "rejected-token"})
    assert response.status_code == 502
    assert client.get("/api/v1/llm/settings", headers=owner).json()["profile"] is None
    assert "отклонил ключ" in response.json()["detail"]
    assert "rejected-token" not in response.text


def test_named_checkpoints_validate_models_and_switch_independently(client, monkeypatch):
    import json
    import httpx

    original = httpx.AsyncClient
    available_models = {"test-flash", "test-pro"}
    called = []

    def respond(request):
        body = json.loads(request.content)
        model = body["model"]
        called.append(model)
        assert request.headers["Authorization"] == "Bearer checkpoint-test-key"
        if model not in available_models:
            return httpx.Response(404, json={"error": {"message": "unknown model"}})
        if not body.get("tools"):
            message = {"content": '{"score":0.95,"comment":"Прогноз подходит."}'}
        else:
            message = {"tool_calls": [{"function": {
                "name": body["tools"][0]["function"]["name"], "arguments": "{}",
            }}]}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    monkeypatch.setattr("backend.core.llm.manager.httpx.AsyncClient", lambda **kw: original(
        transport=httpx.MockTransport(respond), **{k: v for k, v in kw.items() if k != "transport"},
    ))
    owner = user(client, "checkpoint-owner")
    guest = user(client, "checkpoint-guest")
    prefix = "/api/v1/llm"
    payload = {"name": "  Быстрый  ", "provider": "deepseek", "model": "invented-id",
               "api_key": "checkpoint-test-key"}
    missing_name = client.post(prefix + "/checkpoints", headers=owner, json={**payload, "name": " "})
    assert missing_name.status_code == 422 and not called
    assert "checkpoint-test-key" not in missing_name.text
    rejected = client.post(prefix + "/checkpoints", headers=owner, json=payload)
    assert rejected.status_code == 502
    assert "Модель не найдена" in rejected.json()["detail"]
    assert client.get(prefix + "/settings", headers=owner).json()["checkpoints"] == []

    first = client.post(prefix + "/checkpoints", headers=owner, json={**payload, "model": "test-flash"})
    second = client.post(prefix + "/checkpoints", headers=owner, json={
        **payload, "name": "Подробный", "model": "test-pro",
    })
    assert first.status_code == second.status_code == 201
    assert first.json()["name"] == "Быстрый"
    assert "checkpoint-test-key" not in first.text
    flash, pro = first.json()["id"], second.json()["id"]
    assert client.get(prefix + "/settings", headers=owner).json()["active_checkpoint_id"] is None
    assert client.get(prefix + "/settings", headers=guest).json()["checkpoints"] == []
    assert client.put(prefix + "/selection", headers=guest, json={"checkpoint_id": flash}).status_code == 404
    assert client.post(prefix + f"/checkpoints/{flash}/probe", headers=guest).status_code == 404
    assert client.post(prefix + "/checkpoints", headers=owner, json={**payload, "model": "test-pro"}).status_code == 409

    selection = client.put(prefix + "/selection", headers=owner, json={"checkpoint_id": flash})
    assert selection.json()["profile"]["model"] == "test-flash"
    available_models.remove("test-pro")
    assert client.post(prefix + f"/checkpoints/{pro}/probe", headers=owner).status_code == 502
    assert client.put(prefix + "/selection", headers=owner, json={"checkpoint_id": pro}).status_code == 502
    assert client.get(prefix + "/settings", headers=owner).json()["active_checkpoint_id"] == flash

    available_models.add("test-pro")
    selected = client.put(prefix + "/selection", headers=owner, json={"checkpoint_id": pro})
    assert selected.json()["active_checkpoint_id"] == pro

    marker = len(called)
    data = client.get("/api/v1/forecast/demo/series").json()
    task = client.post("/api/v1/forecast/", headers=owner, json={"data": data, "horizon": 2})
    wait(client, "/api/v1/forecast/" + task.json()["task_id"] + "/status",
         owner, {"completed", "waiting_for_human"})
    assert called[marker:] and set(called[marker:]) == {"test-pro"}

    assert client.delete(prefix + f"/checkpoints/{pro}", headers=owner).status_code == 204
    final = client.get(prefix + "/settings", headers=owner).json()
    assert final["active_checkpoint_id"] is None
    assert [row["id"] for row in final["checkpoints"]] == [flash]
