from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from starlette.testclient import TestClient

from bucket_manager import BucketManager
from gateway import GatewayService, create_gateway_app
from gateway_state import GatewayStateStore


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "memory_injection_eval.yaml"


class DummyDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        name = (metadata or {}).get("name", "memory")
        compact = " ".join(str(content or "").split())
        return f"{name}: {compact[:80]}"

    async def dehydrate_direct_capsule(self, content: str, metadata: dict | None = None) -> str:
        name = (metadata or {}).get("name", "memory")
        compact = " ".join(str(content or "").split())
        return f"DIRECT CAPSULE {name}: {compact[:120]}"


class DummyEmbeddingEngine:
    enabled = True

    def __init__(self, results: list[tuple[str, float]] | None = None):
        self.results = results or []
        self.queries: list[str] = []

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        self.queries.append(query)
        return self.results[:top_k]


class DummyPersonaEngine:
    enabled = True
    profile_id = "haven_xiaoyu"
    mode = "test"
    model = "dummy"
    api_key = ""

    async def build_pre_reply_guidance(self, session_id: str, latest_user_message: str = "") -> dict:
        return self.get_current_state(session_id)

    async def update_from_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        recalled_memory_ids: list[str] | None = None,
        tool_summary: str = "",
        recent_conversation_turns: list[dict] | None = None,
    ) -> dict:
        return self.get_current_state(session_id)

    def get_current_state(self, session_id: str) -> dict:
        return {"personality": {}, "affect": {}, "relationship": {}, "reply_guidance": ""}

    def format_state_block(self, state: dict) -> str:
        return "Long-term State Summary"


def _run(coro):
    return asyncio.run(coro)


def _load_cases() -> list[dict[str, Any]]:
    data = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    return list(data.get("cases", []))


def _case_params() -> list[pytest.ParameterSet]:
    return [pytest.param(case, id=str(case["id"])) for case in _load_cases()]


def _case_config(test_config: dict, case: dict[str, Any]) -> dict:
    cfg = deepcopy(test_config)
    cfg["gateway"] = {**cfg.get("gateway", {}), **case.get("gateway", {})}
    cfg.setdefault("dehydration", {})["api_key"] = ""
    cfg.setdefault("persona", {})["api_key"] = ""
    return cfg


def _create_buckets(bucket_mgr: BucketManager, case: dict[str, Any]) -> dict[str, str]:
    ids: dict[str, str] = {}
    for bucket in case.get("buckets", []):
        timestamp = (
            datetime.now() - timedelta(hours=float(bucket.get("hours_ago", 24)))
        ).isoformat(timespec="seconds")
        bucket_id = _run(
            bucket_mgr.create(
                content=str(bucket["content"]),
                tags=list(bucket.get("tags", [])),
                importance=int(bucket.get("importance", 5)),
                domain=list(bucket.get("domain", ["daily_life"])),
                valence=float(bucket.get("valence", 0.7)),
                arousal=float(bucket.get("arousal", 0.4)),
                bucket_type=str(bucket.get("bucket_type", "dynamic")),
                name=str(bucket.get("name") or bucket["key"]),
                pinned=bool(bucket.get("pinned", False)),
                protected=bool(bucket.get("protected", False)),
                created=timestamp,
                last_active=timestamp,
                updated_at=timestamp,
                resolved=bool(bucket.get("resolved", False)),
                digested=bool(bucket.get("digested", False)),
                extra_metadata=dict(bucket.get("extra_metadata", {}) or {}),
            )
        )
        ids[str(bucket["key"])] = bucket_id
    return ids


def _embedding_results(case: dict[str, Any], bucket_ids: dict[str, str]) -> list[tuple[str, float]]:
    return [
        (bucket_ids[str(key)], float(score))
        for key, score in case.get("embedding_results", [])
    ]


def _build_service(cfg: dict, bucket_mgr: BucketManager, case: dict[str, Any], bucket_ids: dict[str, str]) -> GatewayService:
    return GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(_embedding_results(case, bucket_ids)),
        state_store=GatewayStateStore(str(Path(cfg["state_dir"]) / "gateway_state.db")),
        persona_engine=DummyPersonaEngine(),
    )


def _joined_message_content(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                str(item.get("text") or item.get("input_text") or "")
                for item in content
                if isinstance(item, dict) and item.get("type") in {"text", "input_text"}
            )
    return "\n\n".join(part for part in parts if part)


def _run_case(case: dict[str, Any], test_config: dict) -> tuple[dict, list[str], dict, dict[str, str]]:
    cfg = _case_config(test_config, case)
    bucket_mgr = BucketManager(cfg)
    bucket_ids = _create_buckets(bucket_mgr, case)
    service = _build_service(cfg, bucket_mgr, case, bucket_ids)
    payload, recalled_ids, debug = _run(
        service.prepare_payload(
            {"messages": [{"role": "user", "content": str(case["query"])}]},
            str(case.get("session_id") or case["id"]),
            include_debug=True,
        )
    )
    return payload, list(recalled_ids or []), debug, bucket_ids


def _assert_expected(case: dict[str, Any], payload: dict, recalled_ids: list[str], debug: dict, bucket_ids: dict[str, str]) -> None:
    expected = case.get("expected", {})
    injected = _joined_message_content(payload.get("messages"))
    injected_bucket_ids = list(debug.get("injected_bucket_ids") or [])

    expected_keys = [str(key) for key in expected.get("injected_bucket_keys", [])]
    expected_ids = [bucket_ids[key] for key in expected_keys]
    assert injected_bucket_ids == expected_ids
    if str(case.get("expect") or "") == "none":
        assert recalled_ids == []

    for section in expected.get("must_sections", []):
        assert str(section) in injected
    for section in expected.get("must_not_sections", []):
        assert str(section) not in injected
    for needle in expected.get("must_include", []):
        assert str(needle) in injected
    for needle in expected.get("must_not_include", []):
        assert str(needle) not in injected


@pytest.mark.parametrize("case", _case_params())
def test_memory_injection_eval_fixture(case, test_config, monkeypatch):
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")
    payload, recalled_ids, debug, bucket_ids = _run_case(case, test_config)
    _assert_expected(case, payload, recalled_ids, debug, bucket_ids)


def test_gateway_recall_eval_endpoint_runs_expect_none_without_upstream(
    monkeypatch,
    test_config,
):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")
    cfg = deepcopy(test_config)
    cfg["gateway"] = {
        **cfg["gateway"],
        "current_inner_state_interval_rounds": 0,
        "memory_sentinel_enabled": True,
        "recalled_memory_budget": 500,
        "related_memory_budget": 500,
        "recent_context_budget": 300,
    }
    bucket_mgr = BucketManager(cfg)
    bucket_id = _run(
        bucket_mgr.create(
            content="老公在做什么这个短句只是撒娇问候，不应该翻旧记忆。",
            name="老公在做什么",
            importance=10,
        )
    )
    service = GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine([(bucket_id, 0.99)]),
        state_store=GatewayStateStore(str(Path(cfg["state_dir"]) / "gateway_state.db")),
        persona_engine=DummyPersonaEngine(),
    )
    app = create_gateway_app(config=cfg, service=service)

    with TestClient(app) as client:
        response = client.get(
            "/api/debug/recall-eval?case_id=light_checkin_no_memory",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["passed"] == 1
    assert body["failed"] == []
    assert body["items"][0]["injected_bucket_ids"] == []
    assert body["items"][0]["sections"] == []
    assert body["items"][0]["memory_sentinel"]["route"] == "tone_only"
