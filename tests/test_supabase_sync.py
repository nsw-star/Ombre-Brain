import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sync_to_supabase.py"
SPEC = importlib.util.spec_from_file_location("sync_to_supabase", MODULE_PATH)
sync = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = sync
SPEC.loader.exec_module(sync)


def _record(bucket_id, *, source="ombre", last_active="2026-05-04T08:00:00+00:00", **overrides):
    record = {
        "id": bucket_id,
        "title": f"title-{bucket_id}",
        "type": "dynamic",
        "domain": ["数字"],
        "tags": [],
        "content": f"content-{bucket_id}",
        "valence": 0.5,
        "arousal": 0.5,
        "importance": 5,
        "pinned": False,
        "activation_count": 1,
        "created": last_active,
        "last_active": last_active,
        "source": source,
    }
    record.update(overrides)
    return record


def test_plan_pulls_only_chatgpt_authored_remote_updates():
    local = [_record("same", source="ombre", last_active="2026-05-04T08:00:00+00:00")]
    remote = [
        _record("same", source="ombre", last_active="2026-05-04T09:00:00+00:00"),
        _record("new-chatgpt", source="chatgpt", last_active="2026-05-04T09:00:00+00:00"),
        _record("new-ombre", source="ombre", last_active="2026-05-04T09:00:00+00:00"),
    ]

    plan = sync.build_plan(local, remote)

    assert [record["id"] for record in plan.to_pull] == ["new-chatgpt"]


def test_plan_pushes_local_newer_by_last_active_not_synced_at():
    local = [_record("local", last_active="2026-05-04T08:30:00+00:00")]
    remote = [
        _record(
            "local",
            source="ombre",
            last_active="2026-05-04T08:00:00+00:00",
            synced_at="2026-05-04T10:00:00+00:00",
        )
    ]

    plan = sync.build_plan(local, remote)

    assert [record["id"] for record in plan.to_push] == ["local"]
    assert plan.to_pull == []


def test_local_path_for_record_uses_archive_folder_and_readable_filename(tmp_path):
    record = _record(
        "abc123",
        type="archived",
        domain=["恋爱"],
        title="亲密互动模式",
    )

    path = sync.local_path_for_record(record, tmp_path)

    assert path == tmp_path / "archive" / "恋爱" / "亲密互动模式_abc123.md"


def test_record_to_md_preserves_chatgpt_source_and_timezone(tmp_path):
    path = tmp_path / "dynamic" / "数字" / "entry.md"
    record = _record(
        "entry",
        source="chatgpt",
        last_active=datetime(2026, 5, 4, 8, 0, tzinfo=timezone.utc).isoformat(timespec="seconds"),
    )

    sync.record_to_md(record, path)
    parsed = sync.parse_md(path)

    assert parsed["source"] == "chatgpt"
    assert parsed["last_active"].endswith("+00:00")


async def test_bucket_manager_create_accepts_client_id_source_and_timezone(bucket_mgr):
    bucket_id = await bucket_mgr.create(
        content="C 端写入的一条记忆。",
        name="C端记忆",
        domain=["同步"],
        bucket_id="chatgpt_memory_20260504",
        source="chatgpt",
        created="2026-05-04T08:00:00+00:00",
        last_active="2026-05-04T08:00:00+00:00",
    )

    bucket = await bucket_mgr.get(bucket_id)

    assert bucket_id == "chatgpt_memory_20260504"
    assert bucket["metadata"]["source"] == "chatgpt"
    assert bucket["metadata"]["created"].endswith("+00:00")


async def test_bucket_manager_update_preserves_client_source(bucket_mgr):
    bucket_id = await bucket_mgr.create(content="旧内容", name="旧记忆")

    ok = await bucket_mgr.update(
        bucket_id,
        content="新内容",
        source="chatgpt",
        last_active="2026-05-04T09:00:00+00:00",
    )
    bucket = await bucket_mgr.get(bucket_id)

    assert ok is True
    assert bucket["content"] == "新内容"
    assert bucket["metadata"]["source"] == "chatgpt"
    assert bucket["metadata"]["last_active"].endswith("+00:00")
