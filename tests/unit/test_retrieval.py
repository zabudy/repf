import numpy as np

from backend.core.config import Settings
from backend.retrieval.hybrid import CaseIndex, HybridRetriever, reciprocal_rank


async def test_hybrid_filters_then_combines_ranks():
    items = []
    for i, frequency in enumerate(["D", "QS", "D", "D"]):
        items.append({"id": str(i), "metadata": {"task_type": "timeseries", "frequency": frequency},
                      "features": [1, i, 1 + i ** 2], "description": "periodic local case",
                      "best_model": "fit_holt", "arguments": {}, "metrics": {"mae": i}})
    index = CaseIndex.build(items)
    meta = {"task_type": "timeseries", "frequency": "D"}
    assert index.vector("timeseries", np.array([1, 0, 1])).shape == (128,)
    search = HybridRetriever(index, Settings(_env_file=None, QDRANT_URL=None))
    cases, backend = await search.retrieve(meta, np.array([1, 0, 1]), exclude_id="3")
    assert backend == "local_pca"
    assert {row["id"] for row in cases} == {"0", "2"}
    assert len(index.qdrant_points()) == 4
    assert (await search.retrieve({**meta, "frequency": "MS"}, np.array([1, 0, 1])))[0] == []
    assert reciprocal_rank([[3, 2], [4, 2]])[0][0] == 2


async def test_qdrant_query_sends_metadata_filter(monkeypatch):
    import json
    import httpx

    real_client = httpx.AsyncClient
    def reply(request):
        body = json.loads(request.content)
        assert request.url.path == "/collections/repf_cases_v1/points/query"
        assert {"key": "frequency", "match": {"value": "D"}} in body["filter"]["must"]
        assert len(body["query"]) == 128
        return httpx.Response(200, json={"result": {"points": [{"payload": {"case_id": "1"}}]}})
    monkeypatch.setattr("backend.retrieval.hybrid.httpx.AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(reply), **kwargs))
    examples = [{"id": str(i), "metadata": {"task_type": "timeseries", "frequency": "D"},
                 "description": "daily trend", "features": [0.1, i], "best_model": "fit_holt",
                 "arguments": {}, "metrics": {}} for i in range(3)]
    index = CaseIndex.build(examples)
    search = HybridRetriever(index, Settings(_env_file=None, QDRANT_URL="http://qdrant:6333"))
    cases, backend = await search.retrieve({"task_type": "timeseries", "frequency": "D"},
                                           np.array([.1, 1]))
    assert backend == "qdrant"
    assert cases[0]["id"] == "1"
