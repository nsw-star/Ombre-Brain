from pathlib import Path


def test_dashboard_comments_show_time_without_emotion_fields():
    html = Path("dashboard.html").read_text(encoding="utf-8")

    assert "var commentTime = c.original_feel_created || c.created || '';" in html
    assert '<div class="comment-time">' in html

    comments_block = html.split("var commentsHtml = comments.length", 1)[1].split("var commentFormHtml", 1)[0]
    assert "c.valence" not in comments_block
    assert "c.arousal" not in comments_block
