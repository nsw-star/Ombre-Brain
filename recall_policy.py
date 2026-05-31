from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from memory_relevance import (
    MemoryRelevanceOptions,
    content_terms_for_query,
    memory_relevance_options_from_config,
    query_has_explicit_entity_marker,
    query_has_technical_recall_marker,
    recall_admission_decision,
)


CONTEXT_ONLY_SECTIONS = frozenset({"affect_anchor", "favorite_reason", "comment"})
WEAK_RECALL_TOPIC_TERMS = frozenset(
    {
        "进度",
        "偏好",
        "情况",
        "状态",
        "事情",
        "东西",
        "内容",
        "相关",
        "记忆",
        "回忆",
        "总结",
        "记录",
        "查询",
        "搜索",
        "最近",
        "之前",
        "过去",
        "现在",
        "当前",
        "安排",
        "计划",
        "问题",
        "目标",
        "anything",
        "current",
        "find",
        "memory",
        "memories",
        "recent",
        "related",
        "search",
        "something",
        "status",
        "thing",
        "things",
        "topic",
    }
)


@dataclass(frozen=True)
class RecallPolicyDecision:
    admit_direct: bool
    admit_diffused: bool
    seed_allowed: bool
    reason: str
    suppressed: bool
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def admit(self) -> bool:
        return self.admit_direct


class RecallPolicy:
    def __init__(
        self,
        options: MemoryRelevanceOptions | None = None,
        *,
        semantic_threshold: float = 0.72,
        rerank_threshold: float = 0.65,
    ) -> None:
        self.options = options or memory_relevance_options_from_config()
        self.semantic_threshold = _safe_float(semantic_threshold, 0.72)
        self.rerank_threshold = _safe_float(rerank_threshold, 0.65)

    def requires_topic_evidence(self, query: str) -> bool:
        return query_has_explicit_entity_marker(query) or query_has_technical_recall_marker(query)

    def should_enforce_topic_evidence(self, query: str, *, allow_body_chain: bool = False) -> bool:
        return self.requires_topic_evidence(query) and not allow_body_chain

    def specific_query_terms(self, query: str) -> list[str]:
        raw = str(query or "")
        terms = list(content_terms_for_query(raw, self.options))
        terms.extend(re.findall(r"\d+(?:\.\d+)+", raw))
        terms.extend(re.findall(r"[A-Za-z]+[A-Za-z0-9_.:-]*\d[A-Za-z0-9_.:-]*", raw))
        kept = []
        seen = set()
        for term in terms:
            cleaned = str(term or "").strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            if key in WEAK_RECALL_TOPIC_TERMS:
                continue
            if re.fullmatch(r"[a-z0-9_.:-]+", key) and len(key) < 3 and not re.fullmatch(r"\d+(?:\.\d+)+", key):
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]+", cleaned) and len(cleaned) < 2:
                continue
            if any(_term_subsumes(existing.lower(), key) for existing in kept):
                continue
            kept = [existing for existing in kept if not _term_subsumes(key, existing.lower())]
            seen = {existing.lower() for existing in kept}
            seen.add(key)
            kept.append(cleaned)
        return kept

    def moment_has_topic_evidence(self, query: str, moment: dict) -> bool:
        terms = self.specific_query_terms(query)
        if not terms:
            return False
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        fields = " ".join(
            [
                str(moment.get("text") or ""),
                str(meta.get("annotation_summary") or ""),
                _evidence_spans_text(meta.get("evidence_spans")),
                str(meta.get("bucket_name") or ""),
                " ".join(str(tag) for tag in (meta.get("bucket_tags") or []) if str(tag).strip()),
                " ".join(str(item) for item in (meta.get("bucket_domain") or []) if str(item).strip()),
            ]
        ).lower()
        return any(term.lower() in fields for term in terms)

    def bucket_has_topic_evidence(self, query: str, bucket: dict) -> bool:
        terms = self.specific_query_terms(query)
        if not terms:
            return False
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        fields = " ".join(
            [
                str(bucket.get("content") or ""),
                str(meta.get("name") or ""),
                str(meta.get("annotation_summary") or ""),
                _evidence_spans_text(meta.get("evidence_spans")),
                " ".join(str(tag) for tag in (meta.get("tags") or []) if str(tag).strip()),
                " ".join(str(item) for item in (meta.get("domain") or []) if str(item).strip()),
            ]
        ).lower()
        return any(term.lower() in fields for term in terms)

    def node_has_topic_evidence(self, query: str, node: dict) -> bool:
        if "bucket_id" in node or node.get("moment_id"):
            return self.moment_has_topic_evidence(query, node)
        return self.bucket_has_topic_evidence(query, node)

    def allows_moment_context(
        self,
        query: str,
        moment: dict,
        *,
        allow_body_chain: bool = False,
    ) -> bool:
        if not self.should_enforce_topic_evidence(query, allow_body_chain=allow_body_chain):
            return True
        return self.moment_has_topic_evidence(query, moment)

    def allows_bucket_context(
        self,
        query: str,
        bucket: dict,
        *,
        allow_body_chain: bool = False,
    ) -> bool:
        if not self.should_enforce_topic_evidence(query, allow_body_chain=allow_body_chain):
            return True
        return self.bucket_has_topic_evidence(query, bucket)

    def has_strong_score(
        self,
        *,
        semantic_score: float | None = None,
        rerank_score: float | None = None,
    ) -> bool:
        return (
            _safe_float(semantic_score, 0.0) >= self.semantic_threshold
            or _safe_float(rerank_score, 0.0) >= self.rerank_threshold
        )

    def assess(
        self,
        query: str,
        node: dict,
        *,
        has_topic_evidence: bool | None = None,
        semantic_score: float | None = None,
        rerank_score: float | None = None,
        high_confidence_edge: bool = False,
        context_only: bool = False,
    ) -> RecallPolicyDecision:
        if has_topic_evidence is None:
            has_topic_evidence = self.node_has_topic_evidence(query, node)
        debug = {
            "requires_topic_evidence": self.requires_topic_evidence(query),
            "has_topic_evidence": bool(has_topic_evidence),
            "specific_query_terms": self.specific_query_terms(query),
            "semantic_score": _maybe_float(semantic_score),
            "rerank_score": _maybe_float(rerank_score),
            "high_confidence_edge": bool(high_confidence_edge),
            "context_only": bool(context_only),
        }

        if context_only:
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason="context_only_temperature_moment",
                suppressed=True,
                debug=debug,
            )

        base = recall_admission_decision(
            query,
            node,
            self.options,
            semantic_score=semantic_score,
            rerank_score=rerank_score,
            high_confidence_edge=high_confidence_edge,
            semantic_threshold=self.semantic_threshold,
            rerank_threshold=self.rerank_threshold,
        )
        debug["base_reason"] = base.reason

        if not base.admit:
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason=base.reason,
                suppressed=True,
                debug=debug,
            )

        if (
            debug["requires_topic_evidence"]
            and not has_topic_evidence
            and not self.has_strong_score(
                semantic_score=semantic_score,
                rerank_score=rerank_score,
            )
            and not high_confidence_edge
        ):
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason="query_topic_evidence_missing",
                suppressed=True,
                debug=debug,
            )

        return RecallPolicyDecision(
            admit_direct=True,
            admit_diffused=True,
            seed_allowed=True,
            reason=base.reason,
            suppressed=False,
            debug=debug,
        )


def is_context_only_section(section: Any) -> bool:
    return str(section or "") in CONTEXT_ONLY_SECTIONS


def _evidence_spans_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if text:
                parts.append(text)
        elif isinstance(item, str) and item.strip():
            parts.append(item.strip())
    return " ".join(parts)


def _term_subsumes(container: str, contained: str) -> bool:
    if container == contained:
        return True
    if not container or not contained:
        return False
    if not re.search(r"\d", contained):
        return False
    return contained in container


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    number = _maybe_float(value)
    return default if number is None else number
