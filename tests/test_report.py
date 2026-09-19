from html.parser import HTMLParser
import json
from pathlib import Path

from sentinel_evc.events import EventLog
from sentinel_evc.report import render, write_report


FIXTURES = Path(__file__).parent / 'fixtures'


class Tags(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_fixture_event_report_does_not_claim_a_real_run_or_pass():
    events = [json.loads(line) for line in (FIXTURES / 'sample_events.jsonl').read_text(encoding='utf-8').splitlines()]
    result = render('fixture-run', {}, {}, {}, '未执行独立校验', events=events)
    assert 'PASS' not in result
    assert '一次实际运行生成' not in result
    assert '预录' not in result
    assert 'LOG_GAP' in result and 'REVOKE' in result
    assert 'first_dropped_seq' in result
    assert 'submitted / accepted / observed' in result
    outcome = next(event['payload'] for event in events if event['type'] == 'OUTCOME')
    assert f"{outcome['submitted']} / {outcome['accepted']} / {outcome['observed']}" in result
    assert '不是功能安全认证' in result


def test_real_events_show_supplied_cursors_and_verify_failure():
    log = EventLog('report-run')
    log.append('OUTCOME', status='fault', submitted=3, accepted=2, observed=1)
    events = [json.loads(line) for line in log.to_jsonl().splitlines()]
    result = render('report-run', {}, {}, {'event_count': 1, 'tip_hash': log.tip_hash}, 'signature verification failed', events=events)
    assert '3 / 2 / 1' in result
    assert 'signature verification failed' in result
    assert 'PASS' not in result


def test_missing_metrics_are_not_zero_or_performance_claims():
    result = render('partial', {'cases': 2, 'full_check_reduction_pct': 50}, {'details': [{'fault': 'late', 'blocked': False, 'code': 'TRACKING_TUBE', 'cursors': {}}]}, {}, '未校验')
    assert '未记录 / 未记录 / 未记录' in result
    assert '减少' not in result
    assert '50%' not in result
    assert '0 / 0 / 0' not in result
    assert '未阻断' in result
    assert '端到端墙钟时间未记录' in result
    assert '失败回退耗时未记录' in result


def test_all_input_text_is_escaped_and_page_is_offline(tmp_path):
    attack = '\"><script src="https://example.invalid/a.js"></script>'
    sample = {'case': attack, 'kind': attack, 'verdict': attack, 'truly_ok': False,
              'min_margin': -0.01, 'parent_knots': [[0.1, 0.1, 0.3], [0.7, 0.1, 0.3]],
              'child_knots': [[0.1, 0, 0.3], [0.7, 0, 0.3]],
              'obstacle': {'center': [0.4, 0, 0.3], 'radius': 0.05}}
    geo = {'samples': [sample], 'path_b_wrong_release': 7}
    before = json.dumps(geo)
    path = tmp_path / 'report.html'
    write_report(str(path), run_id=attack, geo=geo, faults={}, bundle={}, verify_msg=attack)
    result = path.read_text(encoding='utf-8')
    tags = Tags(result).tags
    assert any(tag == 'svg' for tag, _ in tags)
    assert all(tag not in {'script', 'link', 'iframe'} for tag, _ in tags)
    assert all('src' not in attrs and 'href' not in attrs and not any(k.startswith('on') for k in attrs) for _, attrs in tags)
    assert '&lt;script' in result
    assert "class='num bad'>7" in result
    assert json.dumps(geo) == before
