from backend.core.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.PROJECT_NAME == "REPF"
    assert s.DEFAULT_REGION == "Kazan"
    assert 0 < s.CONFIDENCE_THRESHOLD < 1


def test_env_override(monkeypatch):
    monkeypatch.setenv("PROJECT_NAME", "TEST")
    assert Settings(_env_file=None).PROJECT_NAME == "TEST"
