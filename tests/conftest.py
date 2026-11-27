"""Use a disposable database and never reach an external LLM in pytest."""
import atexit
import os
from pathlib import Path
from tempfile import TemporaryDirectory

test_artifacts = TemporaryDirectory(prefix="repf-tests-")
atexit.register(test_artifacts.cleanup)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{Path(test_artifacts.name) / 'test.db'}"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["WORKER_MODE"] = "local"
os.environ["ADMIN_EMAIL"] = ""
os.environ["ADMIN_PASSWORD"] = ""
os.environ["QDRANT_URL"] = ""
os.environ["HITL_ENABLED"] = "true"
os.environ["JWT_SECRET_KEY"] = "integration-test-signing-key-00000001"
