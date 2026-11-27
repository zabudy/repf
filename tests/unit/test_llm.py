"""Real DeepSeek request and response shapes are tested via httpx transports."""
import json

import httpx
import pytest

from backend.core.config import Settings
from backend.core.forecast import ToolChoice
from backend.core.llm.manager import Critique, LLMManager
from backend.core.llm.tools import validate


@pytest.fixture
def fallback():
    return validate(ToolChoice(name="fit_holt"), "timeseries")


async def test_deepseek_native_tool_call(fallback):
    def handler(request):
        payload = json.loads(request.content)
        assert request.url.host == "api.deepseek.com"
        assert request.headers["Authorization"] == "Bearer test-token"
        assert payload["tool_choice"] == "required"
        assert payload["thinking"] == {"type": "disabled"}
        assert len(payload["tools"]) == 2
        return httpx.Response(200, json={
            "choices": [{"message": {"tool_calls": [
                {"function": {"name": "fit_harmonic", "arguments": '{"harmonics":2}'}},
            ]}}],
            "usage": {"prompt_tokens": 39, "completion_tokens": 11},
        })

    events = []
    async def log(event):
        events.append(event)
    settings = Settings(_env_file=None, LLM_PROVIDER="auto", DEEPSEEK_API_KEY="test-token")
    manager = LLMManager(settings, httpx.MockTransport(handler))
    selected, source = await manager.select("timeseries", ["fit_holt", "fit_harmonic"], {}, fallback, log)
    assert selected.name == "fit_harmonic"
    assert source == "deepseek"
    assert events[0]["input_tokens"] == 39
    assert events[0]["output_tokens"] == 11
    assert not events[0]["estimated_tokens"]
    assert "test-token" not in str(events)


async def test_failed_native_calls_fall_back_to_validated_json(fallback):
    responses = []
    def handler(request):
        payload = json.loads(request.content)
        responses.append(payload)
        if request.url.host == "api.deepseek.com" and "tools" in payload:
            return httpx.Response(429, json={"error": "rate limit"})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"tool_calls": [{"function": {
                "name": "fit_holt", "arguments": {"alpha": "bad"},
            }}]}})
        return httpx.Response(200, json={"choices": [{"message": {"content":
            '{"name":"fit_holt","arguments":{"alpha":0.61}}'}}]})

    events = []
    async def log(event):
        events.append(event)
    settings = Settings(_env_file=None, LLM_PROVIDER="auto", DEEPSEEK_API_KEY="test-token", OLLAMA_URL="http://ollama:11434")
    manager = LLMManager(settings, httpx.MockTransport(handler))
    selected, source = await manager.select("timeseries", ["fit_holt"], {}, fallback, log)
    assert source == "deepseek"
    assert selected.arguments["alpha"] == .61
    assert [e["status"] for e in events] == ["error", "invalid_output", "ok"]
    assert len(responses) == 3
    assert responses[-1]["response_format"] == {"type": "json_object"}


async def test_reject_unregistered_tools_and_invalid_critic(fallback):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"score":2,"comment":"bad"}',
            "tool_calls": [{"function": {"name": "unknown", "arguments": "{}"}}]}}]})

    settings = Settings(_env_file=None, LLM_PROVIDER="auto", DEEPSEEK_API_KEY="test-token")
    manager = LLMManager(settings, httpx.MockTransport(handler))
    events = []
    async def log(event):
        events.append(event)
    choice, source = await manager.select("timeseries", ["fit_holt"], {}, fallback, log)
    assert choice == fallback and source == "fallback"
    rubric = Critique(score=.9, comment="validation passed")
    assert await manager.critique({}, rubric, log) == rubric
    assert all(event["status"] == "invalid_output" for event in events)


def test_registry_rejects_different_types_and_unknown_arguments():
    with pytest.raises(ValueError):
        validate(ToolChoice(name="fit_ridge"), "timeseries")
    with pytest.raises(ValueError):
        validate(ToolChoice(name="fit_holt", arguments={"ridge": 3}), "timeseries")
    with pytest.raises(ValueError):
        ToolChoice.model_validate({"name": "fit_holt", "arguments": {"alpha": "0.4"}})


@pytest.mark.parametrize(("selected", "url", "target"), [
    ("deepseek", "https://other.example/v1", "deepseek"),
    ("compatible", "https://other.example/v1", "compatible"),
    ("auto", "https://other.example/v1", "compatible"),
    ("auto", None, "deepseek"),
    ("ollama", "https://other.example/v1", None),
])
def test_generic_key_is_bound_to_one_provider(selected, url, target):
    settings = Settings(_env_file=None, LLM_PROVIDER=selected, LLM_API_KEY="generic-test",
                        COMPATIBLE_LLM_URL=url, COMPATIBLE_LLM_KEY="compatible-test",
                        DEEPSEEK_API_KEY="deepseek-test", OLLAMA_URL="http://ollama:11434")
    configured = LLMManager(settings).providers
    assert [provider.name for provider in configured if provider.token == "generic-test"] == (
        [target] if target else []
    )
    assert configured[0].name == (target or "ollama")
    assert next(provider for provider in configured if provider.name == "ollama").token is None


@pytest.mark.parametrize("provider", ["compatible", "ollama"])
async def test_native_tool_call_for_alternate_providers(provider, fallback):
    def handler(request):
        body = json.loads(request.content)
        assert body["tools"][0]["function"]["name"] == "fit_holt"
        message = {"tool_calls": [{"function": {
            "name": "fit_holt", "arguments": {"alpha": .51},
        }}]}
        if provider == "ollama":
            assert request.url.path == "/api/chat"
            assert "Authorization" not in request.headers
            return httpx.Response(200, json={"message": message})
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer integration-compatible"
        return httpx.Response(200, json={"choices": [{"message": message}]})

    settings = Settings(_env_file=None, LLM_PROVIDER=provider, LLM_API_KEY="integration-compatible",
                        DEEPSEEK_API_KEY=None, OLLAMA_URL="http://ollama:11434",
                        COMPATIBLE_LLM_URL="https://example.org/v1")
    events = []

    async def record(event):
        events.append(event)

    choice, source = await LLMManager(settings, httpx.MockTransport(handler)).select(
        "timeseries", ["fit_holt"], {}, fallback, record, native_only=True,
    )
    assert choice.arguments["alpha"] == .51
    assert source == provider
    assert len(events) == 1 and events[0]["status"] == "ok"


def test_local_profile_cipher_persists_key_without_leaving_temp_files(tmp_path, monkeypatch):
    from backend.core.llm.profile import effective_settings, encrypt
    from backend.db.models import LLMProfile

    keyfile = tmp_path / "key"
    monkeypatch.setattr("backend.core.llm.profile.KEY_PATH", keyfile)
    settings = Settings(_env_file=None, JWT_SECRET_KEY=None)
    ciphertext = encrypt("test-credential", settings)
    profile = LLMProfile(user_id="owner", provider="deepseek", encrypted_key=ciphertext)
    resolved = effective_settings(settings, profile)
    assert resolved.LLM_API_KEY.get_secret_value() == "test-credential"
    assert "test-credential" not in ciphertext
    assert list(tmp_path.iterdir()) == [keyfile]
