# -*- coding: utf-8 -*-
"""Characterization tests for speedtest.py.

These tests pin down the CURRENT behavior of speedtest.py (including quirks)
as a safety net for future refactors. They must never touch the network.
"""

import json
import os
import re
import socket
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import speedtest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if any test accidentally opens a real socket."""
    def guard(*args, **kwargs):
        raise AssertionError('Network access attempted in characterization '
                             'tests')
    monkeypatch.setattr(socket.socket, 'connect', guard)
    monkeypatch.setattr(socket, 'create_connection', guard)


def make_speedtest(**attrs):
    st = speedtest.Speedtest.__new__(speedtest.Speedtest)
    for name, value in attrs.items():
        setattr(st, name, value)
    return st


# ---------------------------------------------------------------- distance()

def test_distance_same_point_is_zero():
    assert speedtest.distance((0, 0), (0, 0)) == 0


def test_distance_london_paris():
    d = speedtest.distance((51.5, -0.13), (48.85, 2.35))
    assert round(d, 4) == 343.4931


def test_distance_antipodal_on_equator_is_half_circumference():
    d = speedtest.distance((0, 0), (0, 180))
    assert round(d, 4) == 20015.0868
    assert d == pytest.approx(3.141592653589793 * 6371)


def test_distance_is_symmetric():
    a, b = (51.5, -0.13), (48.85, 2.35)
    assert speedtest.distance(a, b) == pytest.approx(speedtest.distance(b, a))


# ----------------------------------------------------- get_closest_servers()

SERVERS = {
    10.0: [{'id': '1'}],
    2.0: [{'id': '2'}],
    5.0: [{'id': '3'}, {'id': '4'}],
}


def _closest_ids(**kwargs):
    st = make_speedtest(servers=dict(SERVERS), closest=[])
    return [s['id'] for s in st.get_closest_servers(**kwargs)]


def test_get_closest_servers_default_limit_returns_all_when_fewer_than_5():
    assert _closest_ids() == ['2', '3', '4', '1']


def test_get_closest_servers_respects_smaller_limit():
    assert _closest_ids(limit=2) == ['2', '3']
    assert _closest_ids(limit=3) == ['2', '3', '4']
    assert _closest_ids(limit=1) == ['2']


def test_get_closest_servers_default_limit_is_5():
    servers = dict((float(i), [{'id': str(i)}]) for i in range(10, 0, -1))
    st = make_speedtest(servers=servers, closest=[])
    assert [s['id'] for s in st.get_closest_servers()] == \
        ['1', '2', '3', '4', '5']


def test_get_closest_servers_appends_to_existing_closest():
    # Characterization (quirk): results are appended to self.closest rather
    # than replacing it, and truncation only triggers on len == limit, so a
    # second call on the same instance duplicates entries past the limit.
    st = make_speedtest(servers=dict(SERVERS), closest=[])
    st.get_closest_servers(limit=2)
    second = st.get_closest_servers(limit=2)
    assert [s['id'] for s in second] == ['2', '3', '2', '3', '4', '1']
    assert second is st.closest


# ---------------------------------------------------------- get_best_server()

class FakeClock(object):
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeResponse(object):
    def __init__(self, status, body):
        self.status = status
        self._body = body

    def read(self, n=None):
        return self._body[:n] if n is not None else self._body


def install_fake_connections(monkeypatch, behaviors):
    """``behaviors`` maps host -> list of per-attempt behaviors, each one of:
    a float latency (seconds, 200 'test=test'), 'error' (raises
    socket.error) or ('bad', latency) (200 with wrong body).
    """
    clock = FakeClock()
    monkeypatch.setattr(speedtest.timeit, 'default_timer', clock)
    calls = []
    attempts = dict((h, list(b)) for h, b in behaviors.items())

    def make_conn(kind):
        class FakeConn(object):
            def __init__(self, host, source_address=None, **kwargs):
                self.host = host
                self.source_address = source_address
                self._response = None

            def request(self, method, path, headers=None):
                calls.append((kind, self.host, method, path,
                              dict(headers or {}), self.source_address))
                behavior = attempts[self.host].pop(0)
                if behavior == 'error':
                    raise socket.error('boom')
                if isinstance(behavior, tuple):
                    clock.now += behavior[1]
                    self._response = FakeResponse(200, b'nope=nope')
                else:
                    clock.now += behavior
                    self._response = FakeResponse(200, b'test=test')

            def getresponse(self):
                return self._response

            def close(self):
                pass
        return FakeConn

    monkeypatch.setattr(speedtest, 'SpeedtestHTTPConnection',
                        make_conn('http'))
    monkeypatch.setattr(speedtest, 'SpeedtestHTTPSConnection',
                        make_conn('https'))
    return calls


def make_best_server_instance(closest):
    results = speedtest.SpeedtestResults(opener=object())
    return make_speedtest(_source_address=None, closest=closest,
                          results=results, _best={})


def test_get_best_server_picks_lowest_average(monkeypatch):
    closest = [
        {'id': 'a', 'url': 'http://a.example/speedtest/upload.php'},
        {'id': 'b', 'url': 'http://b.example/speedtest/upload.php'},
        {'id': 'c', 'url': 'https://c.example/speedtest/upload.php'},
    ]
    calls = install_fake_connections(monkeypatch, {
        'a.example': [0.010, 0.020, 0.030],
        'b.example': [0.005, 0.005, 0.005],
        'c.example': ['error', 'error', 'error'],
    })
    st = make_best_server_instance(closest)

    best = st.get_best_server()

    # Characterization (quirk): the average divides the sum of 3 attempts by
    # 6, so the reported latency is HALF the mean round trip time in ms.
    # b: (0.015 / 6) * 1000 = 2.5 rather than the true mean of 5.0 ms.
    assert best['id'] == 'b'
    assert best['latency'] == 2.5
    assert st.results.ping == 2.5
    assert st.results.server is best
    assert st._best == best
    # Only the winner is mutated with a 'latency' key.
    assert 'latency' not in closest[0]
    assert 'latency' not in closest[2]

    # 3 attempts per server, https URLs use the HTTPS connection class.
    assert [(c[0], c[1]) for c in calls] == (
        [('http', 'a.example')] * 3 + [('http', 'b.example')] * 3 +
        [('https', 'c.example')] * 3
    )
    for kind, host, method, path, headers, source in calls:
        assert method == 'GET'
        assert re.match(r'^/speedtest/latency\.txt\?x=\d+$', path)
        assert headers['User-Agent'] == speedtest.build_user_agent()
        assert source is None
    # Characterization (quirk): the '.%s' % i attempt suffix only appears in
    # the debug log; all three requests to a server use the identical path.
    a_paths = set(c[3] for c in calls if c[1] == 'a.example')
    assert len(a_paths) == 1


def _averages(monkeypatch, behaviors_by_host):
    """Run get_best_server once per host in isolation; return latency."""
    out = {}
    for host, behaviors in behaviors_by_host.items():
        install_fake_connections(monkeypatch, {host: behaviors})
        st = make_best_server_instance(
            [{'id': host, 'url': 'http://%s/speedtest/upload.php' % host}])
        out[host] = st.get_best_server()['latency']
    return out


def test_get_best_server_failure_penalty(monkeypatch):
    lat = _averages(monkeypatch, {
        'allfail.example': ['error', 'error', 'error'],
        'onefail.example': [0.010, 'error', 0.010],
        'badbody.example': [('bad', 0.001), 0.010, 0.010],
    })
    # Each failed attempt contributes 3600 (seconds) to the sum:
    # (3 * 3600 / 6) * 1000 == 1800000.0
    assert lat['allfail.example'] == 1800000.0
    assert lat['onefail.example'] == round((3600.02 / 6) * 1000.0, 3)
    # Wrong body -> 3600 penalty even though the request succeeded.
    assert lat['badbody.example'] == round((3600.02 / 6) * 1000.0, 3)


def test_get_best_server_all_failing_still_returns_a_server(monkeypatch):
    # Characterization: a fully failing server is not excluded; it still
    # "wins" with the penalty latency if it is the only candidate.
    install_fake_connections(monkeypatch, {'x.example': ['error'] * 3})
    st = make_best_server_instance(
        [{'id': 'x', 'url': 'http://x.example/speedtest/upload.php'}])
    best = st.get_best_server()
    assert best['id'] == 'x'
    assert st.results.ping == 1800000.0


def test_get_best_server_tie_last_server_wins(monkeypatch):
    # Characterization (quirk): results are keyed by average latency, so a
    # later server with an identical average overwrites an earlier one.
    install_fake_connections(monkeypatch, {
        'first.example': ['error'] * 3,
        'second.example': ['error'] * 3,
    })
    st = make_best_server_instance([
        {'id': 'first', 'url': 'http://first.example/speedtest/upload.php'},
        {'id': 'second', 'url': 'http://second.example/speedtest/upload.php'},
    ])
    assert st.get_best_server()['id'] == 'second'


def test_get_best_server_no_candidates_raises():
    # closest=[] makes get_best_server call get_closest_servers(); servers is
    # non-empty (so no get_servers() network call) but yields no candidates.
    st = make_best_server_instance([])
    st.servers = {1.0: []}
    with pytest.raises(speedtest.SpeedtestBestServerFailure):
        st.get_best_server()


def test_get_best_server_binds_source_address(monkeypatch):
    calls = install_fake_connections(
        monkeypatch, {'a.example': [0.001] * 3})
    st = make_best_server_instance(
        [{'id': 'a', 'url': 'http://a.example/speedtest/upload.php'}])
    st._source_address = '10.0.0.5'
    st.get_best_server()
    assert set(c[5] for c in calls) == set([('10.0.0.5', 0)])


# --------------------------------------------------------- SpeedtestResults

SERVER = {'id': '1234', 'sponsor': 'ACME', 'name': 'Springfield',
          'd': 12.5}
TIMESTAMP = '2020-01-02T03:04:05.000006Z'


def make_results(**overrides):
    r = speedtest.SpeedtestResults(
        download=93000000.5, upload=12000000.25, ping=21.5,
        server=dict(SERVER), client={'ip': '203.0.113.7', 'isp': 'X'},
        opener=object())
    r.timestamp = TIMESTAMP
    r.bytes_sent = 111
    r.bytes_received = 222
    for k, v in overrides.items():
        setattr(r, k, v)
    return r


EXPECTED_DICT = {
    'download': 93000000.5,
    'upload': 12000000.25,
    'ping': 21.5,
    'server': SERVER,
    'timestamp': TIMESTAMP,
    'bytes_sent': 111,
    'bytes_received': 222,
    'share': None,
    'client': {'ip': '203.0.113.7', 'isp': 'X'},
}


def test_results_defaults_without_network():
    opener = object()
    r = speedtest.SpeedtestResults(opener=opener)
    assert r._opener is opener
    assert (r.download, r.upload, r.ping) == (0, 0, 0)
    assert r.server == {} and r.client == {}
    assert r.bytes_sent == 0 and r.bytes_received == 0
    assert r._share is None
    assert re.match(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$',
                    r.timestamp)


def test_results_dict():
    r = make_results()
    assert r.dict() == EXPECTED_DICT
    assert repr(r) == repr(r.dict())


def test_results_dict_returns_live_references():
    # Characterization: dict() does not copy nested server/client dicts.
    r = make_results()
    assert r.dict()['server'] is r.server
    assert r.dict()['client'] is r.client


def test_results_csv_field_order():
    r = make_results()
    assert r.csv() == ('1234,ACME,Springfield,%s,12.5,21.5,93000000.5,'
                       '12000000.25,,203.0.113.7' % TIMESTAMP)


def test_results_csv_custom_delimiter_and_share():
    r = make_results(_share='http://www.speedtest.net/result/1.png')
    assert r.csv(delimiter=';') == (
        '1234;ACME;Springfield;%s;12.5;21.5;93000000.5;12000000.25;'
        'http://www.speedtest.net/result/1.png;203.0.113.7' % TIMESTAMP)


def test_results_csv_quotes_fields_containing_delimiter():
    r = make_results(server=dict(SERVER, sponsor='ACME, Inc.'))
    assert r.csv().split(',')[1:3] == ['"ACME', ' Inc."']


def test_results_csv_header():
    expected = ('Server ID,Sponsor,Server Name,Timestamp,Distance,Ping,'
                'Download,Upload,Share,IP Address')
    assert speedtest.SpeedtestResults.csv_header() == expected
    assert speedtest.SpeedtestResults.csv_header(delimiter='|') == \
        expected.replace(',', '|')
    # Characterization: "Server ID" contains a space, but space is not the
    # delimiter so csv does not quote it; with delimiter=' ' it is quoted.
    assert speedtest.SpeedtestResults.csv_header(delimiter=' ').startswith(
        '"Server ID" Sponsor "Server Name" ')


def test_results_json_roundtrip():
    r = make_results()
    assert json.loads(r.json()) == r.dict()
    assert '\n' not in r.json()


def test_results_json_pretty_is_indented_and_sorted():
    r = make_results()
    pretty = r.json(pretty=True)
    assert json.loads(pretty) == r.dict()
    assert pretty == json.dumps(EXPECTED_DICT, indent=4, sort_keys=True)
    top_level_keys = re.findall(r'^    "([a-z_]+)":', pretty, re.M)
    assert top_level_keys == sorted(EXPECTED_DICT)


# ----------------------------------------------------------- build_request()

def test_build_request_scheme_defaults_to_http():
    req = speedtest.build_request('://www.speedtest.net/foo')
    assert req.get_full_url().startswith('http://www.speedtest.net/foo?x=')


def test_build_request_secure_uses_https():
    req = speedtest.build_request('://www.speedtest.net/foo', secure=True)
    assert req.get_full_url().startswith('https://www.speedtest.net/foo?x=')


def test_build_request_leaves_explicit_scheme_alone():
    req = speedtest.build_request('http://example.com/a', secure=True)
    # Characterization: secure only applies to ':'-prefixed URLs.
    assert req.get_full_url().startswith('http://example.com/a?x=')


def test_build_request_cache_buster_delimiter():
    plain = speedtest.build_request('http://example.com/a').get_full_url()
    assert re.match(r'^http://example\.com/a\?x=\d+\.0$', plain)

    query = speedtest.build_request(
        'http://example.com/a?b=1').get_full_url()
    assert re.match(r'^http://example\.com/a\?b=1&x=\d+\.0$', query)


def test_build_request_bump_suffix():
    url = speedtest.build_request('http://example.com/a',
                                  bump='7').get_full_url()
    assert re.match(r'^http://example\.com/a\?x=\d+\.7$', url)


def test_build_request_headers():
    req = speedtest.build_request('http://example.com/a')
    # urllib normalizes header names with str.capitalize()
    assert req.get_header('Cache-control') == 'no-cache'
    assert req.get_method() == 'GET'
    # Characterization: despite the docstring, no User-Agent header is
    # added here; the opener's addheaders supplies it.
    assert not req.has_header('User-agent')


def test_build_request_mutates_caller_headers_and_post():
    headers = {'Referer': 'r'}
    req = speedtest.build_request('http://example.com/a', data=b'x=1',
                                  headers=headers)
    # Characterization (quirk): the caller's headers dict is mutated.
    assert headers == {'Referer': 'r', 'Cache-Control': 'no-cache'}
    assert req.get_method() == 'POST'
    assert req.data == b'x=1'


# --------------------------------------------------------- set_mini_server()

class FakeMiniResponse(object):
    def __init__(self, url, body, code=200):
        self._url = url
        self._body = body
        self.code = code

    def read(self):
        return self._body

    def geturl(self):
        return self._url

    def close(self):
        pass


class FakeOpener(object):
    def __init__(self, index_body, upload_bodies=None):
        self.index_body = index_body
        self.upload_bodies = upload_bodies or {}
        self.opened = []

    def open(self, request):
        url = getattr(request, 'get_full_url', lambda: request)()
        self.opened.append(url)
        if isinstance(request, str):
            if url in self.upload_bodies:
                return FakeMiniResponse(url, self.upload_bodies[url])
            raise speedtest.URLError('nope')
        return FakeMiniResponse(url, self.index_body)


@pytest.mark.parametrize('body', [
    b'... uploadExtension: "php" ...',
    b'... upload_extension: "php" ...',
])
def test_set_mini_server_from_extension_in_page(body):
    st = make_speedtest(_opener=FakeOpener(body))
    servers = st.set_mini_server('http://mini.example.com')
    assert servers is st.servers
    assert servers == [{
        'sponsor': 'Speedtest Mini',
        'name': 'mini.example.com',
        'd': 0,
        'url': 'http://mini.example.com/speedtest/upload.php',
        'latency': 0,
        'id': 0,
    }]
    assert re.match(r'^http://mini\.example\.com\?x=\d+\.0$',
                    st._opener.opened[0])


def test_set_mini_server_strips_file_from_path():
    st = make_speedtest(_opener=FakeOpener(b'uploadExtension: "asp"'))
    servers = st.set_mini_server('http://mini.example.com/mini/index.html')
    assert servers[0]['url'] == \
        'http://mini.example.com/mini/speedtest/upload.asp'
    assert servers[0]['name'] == 'mini.example.com'


def test_set_mini_server_trailing_slash_dir():
    # Characterization: a directory URL with a trailing slash has no
    # extension, so it is used as-is and the slash is stripped.
    st = make_speedtest(_opener=FakeOpener(b'uploadExtension: "php"'))
    servers = st.set_mini_server('http://mini.example.com/mini/')
    assert servers[0]['url'] == \
        'http://mini.example.com/mini/speedtest/upload.php'


def test_set_mini_server_probes_upload_extensions():
    base = 'http://mini.example.com'
    opener = FakeOpener(b'no extension here', {
        base + '/speedtest/upload.jsp': b'size=500\n',
    })
    st = make_speedtest(_opener=opener)
    servers = st.set_mini_server(base)
    assert servers[0]['url'] == base + '/speedtest/upload.jsp'
    assert opener.opened[1:] == [base + '/speedtest/upload.%s' % e
                                 for e in ('php', 'asp', 'aspx', 'jsp')]


def test_set_mini_server_invalid():
    st = make_speedtest(_opener=FakeOpener(b'nothing'))
    with pytest.raises(speedtest.InvalidSpeedtestMiniServer):
        st.set_mini_server('http://mini.example.com')


def test_set_mini_server_connect_failure():
    class FailingOpener(object):
        def open(self, request):
            raise speedtest.URLError('down')
    st = make_speedtest(_opener=FailingOpener())
    with pytest.raises(speedtest.SpeedtestMiniConnectFailure):
        st.set_mini_server('http://mini.example.com')


# -------------------------------------------------- shell() arg validation

@pytest.fixture
def run_shell(monkeypatch):
    monkeypatch.setattr(speedtest.signal, 'signal', lambda *a, **k: None)

    def run(*argv):
        monkeypatch.setattr(sys, 'argv', ['speedtest.py'] + list(argv))
        speedtest.shell()
    return run


def test_shell_rejects_no_download_and_no_upload(run_shell):
    with pytest.raises(speedtest.SpeedtestCLIError) as exc:
        run_shell('--no-download', '--no-upload')
    assert str(exc.value) == ('Cannot supply both --no-download and '
                              '--no-upload')


@pytest.mark.parametrize('delim', ['', ';;'])
def test_shell_rejects_bad_csv_delimiter(run_shell, delim):
    with pytest.raises(speedtest.SpeedtestCLIError) as exc:
        run_shell('--csv-delimiter', delim)
    assert str(exc.value) == '--csv-delimiter must be a single character'


def test_main_converts_cli_error_to_system_exit(run_shell, monkeypatch):
    monkeypatch.setattr(sys, 'argv',
                        ['speedtest.py', '--no-download', '--no-upload'])
    with pytest.raises(SystemExit) as exc:
        speedtest.main()
    assert exc.value.code == ('ERROR: Cannot supply both --no-download and '
                              '--no-upload')
