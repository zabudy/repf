"""Make one native Tool Calling smoke test without printing credentials."""
import asyncio

from backend.core.config import get_settings
from backend.core.forecast import ToolChoice
from backend.core.llm.manager import LLMManager
from backend.core.llm.tools import validate


async def main() -> None:
    manager = LLMManager(get_settings())
    manager.providers = [provider for provider in manager.providers if provider.name == "deepseek"]
    if not manager.providers:
        raise SystemExit("Set LLM_PROVIDER=deepseek and LLM_API_KEY in .env")
    calls = []

    async def log(entry: dict) -> None:
        calls.append(entry)
        print({key: entry.get(key) for key in ("provider", "model", "status", "error", "tool_calling",
                                               "input_tokens", "output_tokens", "latency_ms")})

    fallback = validate(ToolChoice(name="fit_holt"), "timeseries")
    choice, source = await manager.select(
        "timeseries", ["fit_holt", "fit_harmonic"], {"horizon": 4, "trend": 1.5}, fallback, log,
    )
    if source != "deepseek" or not any(call["tool_calling"] and call["status"] == "ok" for call in calls):
        raise SystemExit("DeepSeek native Tool Calling was not confirmed")
    print("Validated tool:", choice.name)


if __name__ == "__main__":
    asyncio.run(main())
