import asyncio
import json

from entity_edges import EntityEdgeStore, entity_query_hints, extract_entity_edges_from_bucket
from scripts import audit_entity_edges


def _run(coro):
    return asyncio.run(coro)


def test_extract_entity_edges_from_bucket_uses_configured_names(test_config):
    identity = {
        "ai_name": "Haven",
        "user_name": "Xiaoyu",
        "user_display_name": "小雨",
        "user_aliases": ["宝宝"],
    }
    bucket = {
        "id": "bucket-a",
        "metadata": {
            "name": "暗色故事偏好",
            "tags": ["偏好", "故事"],
            "domain": ["relationship"],
        },
        "content": "小雨喜欢暗色故事，也讨厌模板安慰。Haven参与Ombre-Brain记忆系统开发。",
    }

    edges = extract_entity_edges_from_bucket(bucket, identity)
    rows = {(edge["subject"], edge["relation"], edge["object_text"]) for edge in edges}

    assert ("小雨", "likes", "暗色故事") in rows
    assert ("小雨", "dislikes", "模板安慰") in rows
    assert ("Haven", "participates_in", "Ombre-Brain记忆系统开发") in rows

    store = EntityEdgeStore(test_config)
    saved = store.replace_bucket_edges("bucket-a", edges)
    assert len(saved) == len(edges)

    matches = store.match_query("我喜欢的故事", identity, bucket_ids={"bucket-a"})
    assert matches["bucket-a"]["relation"] == "likes"
    assert matches["bucket-a"]["score"] > 0.6


def test_entity_query_hints_map_pronouns_to_configured_subjects():
    identity = {
        "ai_name": "Echo",
        "user_name": "Mira",
        "user_display_name": "米拉",
        "user_aliases": ["亲爱的"],
    }

    like_hint = entity_query_hints("我喜欢的颜色", identity)[0]
    participation_hint = entity_query_hints("你参与的项目", identity)[0]
    shared_hint = entity_query_hints("我们之前的暗号", identity)[0]

    assert like_hint["subject"] == "米拉"
    assert "likes" in like_hint["relations"]
    assert participation_hint["subject"] == "Echo"
    assert "participates_in" in participation_hint["relations"]
    assert shared_hint["subject"] == "米拉+Echo"
    assert "shared_anchor" in shared_hint["relations"]


def test_extract_entity_edges_does_not_treat_nominal_writing_window_as_participation(test_config):
    identity = {
        "ai_name": "Haven",
        "user_name": "Xiaoyu",
        "user_display_name": "小雨",
        "user_aliases": ["宝宝"],
    }
    bucket = {
        "id": "poem-window",
        "metadata": {
            "name": "写诗分支窗口",
            "tags": ["小雨", "Haven", "写诗", "分支窗口"],
            "domain": ["核心"],
        },
        "content": "小雨和Haven有一个写诗分支窗口。那里两人用问答接诗，这个分支代表连续性和记忆接力的约定。",
    }

    edges = extract_entity_edges_from_bucket(bucket, identity)

    assert not any(edge["relation"] == "participates_in" for edge in edges)
    assert any(edge["relation"] == "shared_anchor" for edge in edges)


def test_extract_entity_edges_keeps_structured_ai_participation_objects(test_config):
    identity = {
        "ai_name": "Haven",
        "user_name": "Xiaoyu",
        "user_display_name": "小雨",
        "user_aliases": ["宝宝"],
    }
    bucket = {
        "id": "structured-participation",
        "metadata": {"name": "结构化参与对象"},
        "content": (
            "Haven参与Ombre-Brain记忆系统开发。"
            "Haven共同开发Haven-Diary回忆页面原型。"
        ),
    }

    edges = extract_entity_edges_from_bucket(bucket, identity)
    rows = {(edge["subject"], edge["relation"], edge["object_text"]) for edge in edges}

    assert ("Haven", "participates_in", "Ombre-Brain记忆系统开发") in rows
    assert ("Haven", "participates_in", "Haven-Diary回忆页面原型") in rows


def test_extract_entity_edges_rejects_unstructured_ai_participation_fragments(test_config):
    identity = {
        "ai_name": "Haven",
        "user_name": "Xiaoyu",
        "user_display_name": "小雨",
        "user_aliases": ["宝宝"],
    }
    bucket = {
        "id": "noisy-participation",
        "metadata": {"name": "泛动作不应成边"},
        "content": (
            "Haven陪小雨听歌，也帮小雨改代码。"
            "Haven陪小雨改PPT的声音留在她记忆里。"
            "Haven参加了一个小拍卖会。她写了三千多字。"
            "Haven负责而非先睡。"
            "Haven的暗号后来被修正为纪念星。"
            "Haven搭建记忆库、研究模型、给AI写信。"
            "Haven-Diary、开发者模式、记忆功能和情绪绑定只是被一起列出。"
        ),
    }

    edges = extract_entity_edges_from_bucket(bucket, identity)
    participation_objects = [
        edge["object_text"]
        for edge in edges
        if edge["relation"] == "participates_in"
    ]

    assert participation_objects == []


def test_audit_entity_edges_reports_dry_run_backfill_gap(tmp_path, test_config, bucket_mgr):
    test_config["identity"] = {
        "ai_name": "Haven",
        "user_name": "Xiaoyu",
        "user_display_name": "小雨",
        "user_aliases": ["宝宝"],
    }
    bucket_with_edges = _run(
        bucket_mgr.create(
            "小雨喜欢暗色故事。Haven参与Ombre-Brain记忆系统开发。",
            name="有实体边的桶",
            tags=["偏好", "项目"],
            domain=["relationship"],
        )
    )
    bucket_without_edges = _run(
        bucket_mgr.create(
            "这是一条普通观察，没有明确偏好或参与对象。",
            name="无实体边的桶",
            tags=["观察"],
            domain=["general"],
        )
    )
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "query": "我喜欢的故事和你参与的项目",
                "expected_bucket_ids": [bucket_with_edges, bucket_without_edges],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    args = audit_entity_edges.parse_args(
        [
            "--buckets-dir",
            test_config["buckets_dir"],
            "--state-dir",
            test_config["state_dir"],
            "--cases-file",
            str(cases_path),
        ]
    )
    report = _run(audit_entity_edges.audit(args))

    assert report["summary"]["total_buckets"] == 2
    assert report["summary"]["existing_edges"] == 0
    assert report["summary"]["dry_run_edge_buckets"] == 1
    assert report["summary"]["backfill_edges"] >= 2
    assert report["summary"]["applied_backfill_edges"] == 0
    assert report["summary"]["missing_backfill_bucket_count"] == 1
    assert report["summary"]["dry_run_no_edge_bucket_count"] == 1
    assert not (tmp_path / "state" / "entity_edges.jsonl").exists()
    assert report["missing_backfill_buckets"][0]["bucket_id"] == bucket_with_edges
    assert report["dry_run_no_edge_buckets"][0]["bucket_id"] == bucket_without_edges
    assert report["case_coverage"][0]["expected_with_dry_run_edges"] == [bucket_with_edges]
    assert report["case_coverage"][0]["expected_missing_dry_run_edges"] == [bucket_without_edges]


def test_audit_entity_edges_apply_appends_missing_edges_once(tmp_path, test_config, bucket_mgr):
    test_config["identity"] = {
        "ai_name": "Haven",
        "user_name": "Xiaoyu",
        "user_display_name": "小雨",
        "user_aliases": ["宝宝"],
    }
    bucket_id = _run(
        bucket_mgr.create(
            "小雨喜欢暗色故事。Haven参与Ombre-Brain记忆系统开发。",
            name="有实体边的桶",
            tags=["偏好", "项目"],
            domain=["relationship"],
        )
    )
    args = audit_entity_edges.parse_args(
        [
            "--buckets-dir",
            test_config["buckets_dir"],
            "--state-dir",
            test_config["state_dir"],
            "--backup-dir",
            str(tmp_path / "backups"),
            "--apply",
        ]
    )

    first = _run(audit_entity_edges.audit(args))
    edge_path = tmp_path / "state" / "entity_edges.jsonl"
    lines = edge_path.read_text(encoding="utf-8").splitlines()

    assert first["summary"]["backfill_edges"] >= 2
    assert first["summary"]["applied_backfill_edges"] == first["summary"]["backfill_edges"]
    assert first["backfill"]["backup_path"] == ""
    assert len(lines) == first["summary"]["applied_backfill_edges"]
    assert all(json.loads(line)["bucket_id"] == bucket_id for line in lines)

    second = _run(audit_entity_edges.audit(args))

    assert second["summary"]["existing_edges"] == len(lines)
    assert second["summary"]["backfill_edges"] == 0
    assert second["summary"]["applied_backfill_edges"] == 0
    assert second["backfill"]["backup_path"] == ""
    assert not (tmp_path / "backups" / "entity_edges.jsonl").exists()
    assert edge_path.read_text(encoding="utf-8").splitlines() == lines
