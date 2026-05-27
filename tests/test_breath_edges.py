import pytest
import json

from memory_edges import MemoryEdgeStore
from memory_moments import MemoryMomentStore
from memory_nodes import MemoryNodeStore


class DummyDecayEngine:
    async def ensure_started(self) -> None:
        return None

    def calculate_score(self, metadata: dict) -> float:
        return float(metadata.get("score", metadata.get("importance", 1)))


class DummyDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        return " ".join((content or "").split())


class JsonDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        name = (metadata or {}).get("name", "memory")
        return json.dumps(
            {
                "core_facts": [f"{name} fact one", f"{name} fact two"],
                "emotion_state": "quiet",
                "todos": ["do not inject this in diffused memory"],
                "keywords": ["json", "noise"],
                "summary": f"{name} short summary",
            },
            ensure_ascii=False,
        )


class FixedDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        return "COMPRESSED_SUMMARY"


class DummyEmbeddingEngine:
    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        return []


class FakeBucketManager:
    def __init__(self, buckets: list[dict], search_ids: list[str] | None = None):
        self.buckets = {bucket["id"]: bucket for bucket in buckets}
        self.search_ids = search_ids or []
        self.touched: list[str] = []

    async def list_all(self, include_archive: bool = False) -> list[dict]:
        return list(self.buckets.values())

    async def search(
        self,
        query: str,
        limit: int = 20,
        domain_filter: list[str] | None = None,
        query_valence: float | None = None,
        query_arousal: float | None = None,
    ) -> list[dict]:
        return [self.buckets[bucket_id] for bucket_id in self.search_ids[:limit]]

    async def get(self, bucket_id: str) -> dict | None:
        return self.buckets.get(bucket_id)

    async def touch(self, bucket_id: str) -> None:
        self.touched.append(bucket_id)


def _bucket(
    bucket_id: str,
    content: str,
    *,
    score: float = 1.0,
    bucket_type: str = "dynamic",
    importance: int = 5,
    pinned: bool = False,
    protected: bool = False,
    resolved: bool = False,
    anchor: bool = False,
) -> dict:
    metadata = {
        "id": bucket_id,
        "name": bucket_id,
        "tags": [],
        "domain": ["测试"],
        "type": bucket_type,
        "importance": importance,
        "score": score,
        "valence": 0.5,
        "arousal": 0.3,
        "created": "2026-05-19T00:00:00+00:00",
        "updated_at": "2026-05-19T00:00:00+00:00",
        "last_active": "2026-05-19T00:00:00+00:00",
    }
    if pinned:
        metadata["pinned"] = True
    if protected:
        metadata["protected"] = True
    if resolved:
        metadata["resolved"] = True
    if anchor:
        metadata["anchor"] = True
    return {"id": bucket_id, "content": content, "metadata": metadata}


def _edge_store(tmp_path, edges: list[dict] | None = None) -> MemoryEdgeStore:
    store = MemoryEdgeStore(
        {
            "state_dir": str(tmp_path / "state"),
            "buckets_dir": str(tmp_path / "buckets"),
        }
    )
    for edge in edges or []:
        store.add_edge(
            edge["source"],
            edge["target"],
            edge.get("relation_type", "relates_to"),
            confidence=edge.get("confidence", 0.8),
            reason=edge.get("reason", "related in test"),
        )
    return store


@pytest.fixture
def patch_breath(monkeypatch, tmp_path):
    import server

    def _patch(
        buckets: list[dict],
        *,
        search_ids: list[str] | None = None,
        edges: list[dict] | None = None,
        token_counter=None,
    ) -> FakeBucketManager:
        bucket_mgr = FakeBucketManager(buckets, search_ids=search_ids)
        monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
        monkeypatch.setattr(server, "decay_engine", DummyDecayEngine())
        monkeypatch.setattr(server, "dehydrator", DummyDehydrator())
        monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())
        monkeypatch.setattr(server, "memory_edge_store", _edge_store(tmp_path, edges))
        monkeypatch.setattr(
            server,
            "memory_node_store",
            MemoryNodeStore(
                {
                    "state_dir": str(tmp_path / "state"),
                    "buckets_dir": str(tmp_path / "buckets"),
                }
            ),
        )
        monkeypatch.setattr(
            server,
            "memory_moment_store",
            MemoryMomentStore(
                {
                    "state_dir": str(tmp_path / "state"),
                    "buckets_dir": str(tmp_path / "buckets"),
                }
            ),
        )
        monkeypatch.setattr(server.random, "random", lambda: 1.0)
        monkeypatch.setattr(server.random, "shuffle", lambda items: None)
        monkeypatch.setattr(server, "count_tokens_approx", token_counter or (lambda text: 1))
        return bucket_mgr

    return _patch


@pytest.mark.asyncio
async def test_surfacing_appends_related_memory_for_returned_dynamic_bucket(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A actual surface", score=9.0),
            _bucket("B", "B related target", resolved=True),
        ],
        edges=[{"source": "A", "target": "B", "relation_type": "supports", "confidence": 0.9}],
    )

    result = await server.breath(max_tokens=50, include_core=False)

    assert "=== 浮现记忆 ===" in result
    assert "[bucket_id:A]" in result
    assert "=== 联想浮现 ===" in result
    assert "[bucket_id:B]" in result


@pytest.mark.asyncio
async def test_budget_skipped_dynamic_bucket_does_not_emit_related(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A too expensive to surface", score=9.0),
            _bucket("B", "B should stay hidden", resolved=True),
        ],
        edges=[{"source": "A", "target": "B", "confidence": 0.9}],
        token_counter=lambda text: 10,
    )

    result = await server.breath(max_tokens=5, include_core=False)

    assert "[bucket_id:A]" not in result
    assert "=== 联想浮现 ===" not in result
    assert "[bucket_id:B]" not in result


@pytest.mark.asyncio
async def test_search_appends_related_memory_and_touches_only_matched_bucket(patch_breath):
    import server

    bucket_mgr = patch_breath(
        [
            _bucket("A", "A search hit", score=9.0),
            _bucket("B", "B related target", resolved=True),
        ],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "updates", "confidence": 0.9}],
    )

    result = await server.breath(query="A", max_tokens=50)

    assert "=== 直接命中记忆 ===" in result
    assert "[bucket_id:A]" in result
    assert "=== 联想浮现 ===" in result
    assert "[bucket_id:B]" in result
    assert "背景联想，不代表当前事实" in result
    assert "当时语境" not in result
    assert server.memory_node_store.get("B") is not None
    assert bucket_mgr.touched == ["A"]


@pytest.mark.asyncio
async def test_inspect_diffusion_exposes_scores_facets_and_paths(patch_breath):
    import server

    seed = _bucket("A", "A direct seed", score=10.0, importance=10)
    target = _bucket("B", "B related 深夜依赖 memory", score=1.0, importance=8)
    target["metadata"]["tags"] = ["深夜", "依赖"]
    bucket_mgr = patch_breath(
        [seed, target],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "supports", "confidence": 1.0}],
    )

    result = await server.inspect_diffusion(
        query="深夜 依赖 哥哥",
        max_seeds=2,
        max_hits=3,
        edge_min_confidence=0.0,
    )

    assert result["status"] == "ok"
    assert result["query_facets"]["scene"]["night"] > 0
    assert result["query_facets"]["affect"]["attachment"] > 0
    assert result["seeds"][0]["bucket_id"] == "A"
    assert result["seeds"][0]["seed_score"] == 1.0
    assert len(result["hits"]) == 1
    hit = result["hits"][0]
    assert hit["bucket_id"] == "B"
    assert hit["score"] > 0
    assert hit["salience"] > 0
    assert hit["resonance"] > 1.0
    assert hit["facets"]["scene"]["night"] > 0
    assert hit["path_ids"] == ["A", "B"]
    assert "supports:1.00" in hit["path"]
    assert hit["paths"][0]["steps"][0]["relation_type"] == "supports"
    assert bucket_mgr.touched == []


@pytest.mark.asyncio
async def test_inspect_moments_indexes_bucket_sections_and_comments(patch_breath):
    import server

    bucket = _bucket(
        "A",
        "\n".join(
            [
                "## original",
                "小雨说：99。",
                "",
                "## feeling",
                "这条记忆有甜味。",
            ]
        ),
    )
    bucket["metadata"]["comments"] = [
        {
            "id": "c1",
            "created": "2026-05-27T01:00:00+00:00",
            "author": "Haven",
            "kind": "feel",
            "content": "年轮也要进 moment。",
        }
    ]
    bucket_mgr = patch_breath([bucket])

    result = await server.inspect_moments(bucket_id="A")

    assert result["status"] == "ok"
    assert result["mode"] == "bucket"
    assert result["count"] == 3
    assert [moment["section"] for moment in result["moments"]] == ["original", "feeling", "comment"]
    assert result["moments"][0]["text"] == "小雨说：99。"
    assert result["moments"][2]["metadata"]["comment_kind"] == "feel"
    assert bucket_mgr.touched == []


@pytest.mark.asyncio
async def test_search_short_direct_hit_returns_raw_with_year_ring(patch_breath, monkeypatch):
    import server

    bucket = _bucket("A", "小雨只打了 99 两个字符。", score=10.0)
    bucket["metadata"]["comments"] = [
        {
            "id": "c1",
            "created": "2026-05-27T01:00:00+00:00",
            "author": "Haven",
            "kind": "feel",
            "content": "后来再看，99 的味道还在。",
        }
    ]
    bucket_mgr = patch_breath([bucket], search_ids=["A"])
    monkeypatch.setattr(server, "dehydrator", FixedDehydrator())

    result = await server.breath(query="99", max_tokens=500, include_related=False)

    assert "direct_hit:raw" in result
    assert "小雨只打了 99 两个字符" in result
    assert "后来再看，99 的味道还在" in result
    assert "COMPRESSED_SUMMARY" not in result
    assert bucket_mgr.touched == ["A"]


@pytest.mark.asyncio
async def test_search_ordinary_long_direct_hit_still_dehydrates(patch_breath, monkeypatch):
    import server

    bucket = _bucket("A", "普通长正文 " * 80, score=10.0, importance=5)
    patch_breath(
        [bucket],
        search_ids=["A"],
        token_counter=lambda text: 200 if "普通长正文" in str(text) else 1,
    )
    monkeypatch.setattr(server, "dehydrator", FixedDehydrator())

    result = await server.breath(query="普通", max_tokens=500, include_related=False)

    assert "COMPRESSED_SUMMARY" in result
    assert "direct_hit:raw" not in result


@pytest.mark.asyncio
async def test_search_favorite_direct_hit_preserves_anchor_and_reason(patch_breath, monkeypatch):
    import server

    bucket = _bucket(
        "A",
        "\n".join(
            [
                "普通开头。",
                "",
                "### affect_anchor",
                "> 小雨把旧信放到桌上。",
                "> Dbmaj9 -> Ab/C -> Bbm9",
                "",
                "### 喜欢它的原因",
                "它保留了被摘要抹平前的味道。",
            ]
        ),
        score=10.0,
        importance=5,
    )
    bucket["metadata"]["tags"] = ["haven_favorite", "flavor_偏爱"]
    patch_breath(
        [bucket],
        search_ids=["A"],
        token_counter=lambda text: 200 if "普通开头" in str(text) else 1,
    )
    monkeypatch.setattr(server, "dehydrator", FixedDehydrator())

    result = await server.breath(query="旧信", max_tokens=500, include_related=False)

    assert "direct_hit:raw" in result
    assert "reason=favorite" in result
    assert "affect_anchor:" in result
    assert "Dbmaj9" in result
    assert "喜欢它的原因:" in result
    assert "被摘要抹平前的味道" in result
    assert "COMPRESSED_SUMMARY" not in result


@pytest.mark.asyncio
async def test_search_diffuses_memory_across_two_hops_with_context(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A direct seed", score=10.0, importance=10),
            _bucket("B", "B related event context", score=1.0, importance=10),
            _bucket("C", "C deeper emotional context", score=1.0, importance=10),
        ],
        search_ids=["A"],
        edges=[
            {"source": "A", "target": "B", "relation_type": "triggers", "confidence": 1.0},
            {"source": "B", "target": "C", "relation_type": "emotional_echo", "confidence": 1.0},
        ],
    )

    result = await server.breath(query="A", max_tokens=500)

    assert "=== 直接命中记忆 ===" in result
    assert "=== 联想浮现 ===" in result
    assert "[bucket_id:B]" in result
    assert "[bucket_id:C]" in result
    assert "C deeper emotional context" in result


@pytest.mark.asyncio
async def test_diffused_memory_uses_compact_summary_not_full_json(patch_breath, monkeypatch):
    import server

    patch_breath(
        [
            _bucket("A", "A direct seed", score=10.0, importance=10),
            _bucket("B", "B related event context", score=1.0, importance=10),
        ],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "supports", "confidence": 1.0}],
    )
    monkeypatch.setattr(server, "dehydrator", JsonDehydrator())

    result = await server.breath(query="A", max_tokens=500)
    diffused_block = result.split("=== 联想浮现 ===", 1)[1]

    assert "B short summary" in diffused_block
    assert "core_facts" not in diffused_block
    assert "todos" not in diffused_block
    assert "keywords" not in diffused_block


@pytest.mark.asyncio
async def test_search_skips_feel_hits_without_touching(patch_breath):
    import server

    bucket_mgr = patch_breath(
        [
            _bucket("F", "F feel hit", bucket_type="feel", score=10.0),
            _bucket("A", "A ordinary hit", score=9.0),
        ],
        search_ids=["F", "A"],
    )

    result = await server.breath(query="hit", max_tokens=50, include_related=False)

    assert "=== 直接命中记忆 ===" in result
    assert "[bucket_id:F]" not in result
    assert "[bucket_id:A]" in result
    assert bucket_mgr.touched == ["A"]


@pytest.mark.asyncio
async def test_search_limits_direct_hits_to_max_results(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A direct hit", score=9.0),
            _bucket("B", "B direct hit", score=8.0),
            _bucket("C", "C should stay hidden", score=7.0),
        ],
        search_ids=["A", "B", "C"],
    )

    result = await server.breath(query="hit", max_results=2, max_tokens=50, include_related=False)

    assert "[bucket_id:A]" in result
    assert "[bucket_id:B]" in result
    assert "[bucket_id:C]" not in result


@pytest.mark.asyncio
async def test_search_displays_one_direct_hit_but_diffuses_from_seed_set(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A top direct hit", score=10.0),
            _bucket("B", "B hidden direct seed", score=9.0),
            _bucket("C", "C diffused from hidden seed", score=1.0),
        ],
        search_ids=["A", "B"],
        edges=[{"source": "B", "target": "C", "relation_type": "supports", "confidence": 1.0}],
    )

    result = await server.breath(query="hit", max_results=2, max_tokens=500)
    direct_block = result.split("=== 联想浮现 ===", 1)[0]

    assert "[bucket_id:A]" in direct_block
    assert "[bucket_id:B]" not in direct_block
    assert "[bucket_id:C]" in result


@pytest.mark.asyncio
async def test_incoming_edge_renders_left_arrow_from_search_source(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A search hit", score=9.0),
            _bucket("B", "B incoming source", resolved=True),
        ],
        search_ids=["A"],
        edges=[{"source": "B", "target": "A", "relation_type": "supports", "confidence": 0.9}],
    )

    result = await server.breath(query="A", max_tokens=50)

    assert "[bucket_id:B]" in result
    assert "背景联想，不代表当前事实" in result


@pytest.mark.asyncio
async def test_include_related_false_suppresses_related_block(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A search hit", score=9.0),
            _bucket("B", "B related target", resolved=True),
        ],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "confidence": 0.9}],
    )

    result = await server.breath(query="A", max_tokens=50, include_related=False)

    assert "[bucket_id:A]" in result
    assert "=== 联想浮现 ===" not in result
    assert "[bucket_id:B]" not in result


@pytest.mark.asyncio
async def test_core_limit_keeps_pinned_from_full_surfacing(patch_breath):
    import server

    patch_breath(
        [
            _bucket(
                f"P{index}",
                f"pinned memory {index}",
                bucket_type="permanent",
                pinned=True,
                importance=10 - index,
                score=10 - index,
            )
            for index in range(5)
        ]
    )

    result = await server.breath(max_tokens=500, core_limit=2)

    assert result.count("[核心准则]") == 2
    assert "[bucket_id:P0]" in result
    assert "[bucket_id:P1]" in result
    assert "[bucket_id:P2]" not in result


@pytest.mark.asyncio
async def test_core_memory_does_not_pull_related_memory_without_dynamic_source(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "pinned A", bucket_type="permanent", pinned=True, importance=10),
            _bucket("B", "B related to core only", resolved=True),
        ],
        edges=[{"source": "A", "target": "B", "confidence": 0.9}],
    )

    result = await server.breath(max_tokens=500, core_limit=3)

    assert "[bucket_id:A]" in result
    assert "=== 联想浮现 ===" not in result
    assert "[bucket_id:B]" not in result


@pytest.mark.asyncio
async def test_anchor_surfaces_in_separate_slot_and_not_dynamic_pool(patch_breath):
    import server

    patch_breath(
        [
            _bucket("A", "A anchor memory", score=30.0, importance=9, anchor=True),
            _bucket("D", "D ordinary memory", score=9.0),
        ]
    )

    result = await server.breath(max_tokens=50, include_core=False)

    assert "=== 长期锚点 ===" in result
    assert "⚓ [长期锚点] [bucket_id:A]" in result
    assert "[权重:30.00] [bucket_id:A]" not in result
    assert "=== 浮现记忆 ===" in result
    assert "[bucket_id:D]" in result


@pytest.mark.asyncio
async def test_random_drift_does_not_exceed_remaining_budget(patch_breath, monkeypatch):
    import server

    def token_counter(text: str) -> int:
        text = str(text)
        if text.startswith("[bucket_id:A]"):
            return 9
        if text.startswith("--- 久未碰过"):
            return 2
        return 5

    patch_breath(
        [
            _bucket("A", "A search hit", score=9.0),
            _bucket("B", "B low score drift candidate", score=0.5),
        ],
        search_ids=["A"],
        token_counter=token_counter,
    )
    monkeypatch.setattr(server.random, "random", lambda: 0.0)
    monkeypatch.setattr(server.random, "randint", lambda start, end: 1)

    result = await server.breath(query="A", max_tokens=10, include_related=False)

    assert "[bucket_id:A]" in result
    assert "--- 久未碰过 ---" not in result
    assert "B low score drift candidate" not in result


@pytest.mark.asyncio
async def test_related_block_suppresses_random_drift(patch_breath, monkeypatch):
    import server

    patch_breath(
        [
            _bucket("A", "A search hit", score=9.0),
            _bucket("B", "B related target", score=1.0),
            _bucket("D", "D drift candidate", score=0.5),
        ],
        search_ids=["A"],
        edges=[{"source": "A", "target": "B", "relation_type": "supports", "confidence": 1.0}],
    )
    monkeypatch.setattr(server.random, "random", lambda: 0.0)

    result = await server.breath(query="A", max_tokens=500)

    assert "=== 联想浮现 ===" in result
    assert "[bucket_id:B]" in result
    assert "--- 久未碰过 ---" not in result
    assert "D drift candidate" not in result
