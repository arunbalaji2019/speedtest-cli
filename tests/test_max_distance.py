# -*- coding: utf-8 -*-
"""Offline tests for the --max-distance server filter."""

import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import speedtest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise AssertionError('Network access attempted in offline tests')
    monkeypatch.setattr(socket.socket, 'connect', guard)
    monkeypatch.setattr(socket, 'create_connection', guard)


SERVERS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<settings><servers>
<server url="http://a/upload.php" lat="0" lon="0" name="A" country="X"
 cc="X" sponsor="SA" id="1" host="a:8080" />
<server url="http://b/upload.php" lat="0" lon="1" name="B" country="X"
 cc="X" sponsor="SB" id="2" host="b:8080" />
<server url="http://c/upload.php" lat="0" lon="10" name="C" country="X"
 cc="X" sponsor="SC" id="3" host="c:8080" />
</servers></settings>
"""

D_B = speedtest.distance((0.0, 0.0), (0.0, 1.0))    # ~111.19 km
D_C = speedtest.distance((0.0, 0.0), (0.0, 10.0))   # ~1111.95 km


class FakeStream(object):
    def __init__(self, data):
        self._chunks = [data, b'']

    def read(self, n):
        return self._chunks.pop(0) if self._chunks else b''

    def close(self):
        pass


class FakeResponse(object):
    code = 200

    def close(self):
        pass


@pytest.fixture
def st(monkeypatch):
    monkeypatch.setattr(speedtest, 'catch_request',
                        lambda request, opener=None: (FakeResponse(), False))
    monkeypatch.setattr(speedtest, 'get_response_stream',
                        lambda uh: FakeStream(SERVERS_XML))
    s = speedtest.Speedtest.__new__(speedtest.Speedtest)
    s.servers = {}
    s.config = {'threads': {'download': 4}, 'ignore_servers': []}
    s.lat_lon = (0.0, 0.0)
    s._secure = False
    s._opener = None
    return s


def ids(servers):
    return sorted(int(a['id']) for lst in servers.values() for a in lst)


def test_no_max_distance_keeps_all_servers(st):
    assert ids(st.get_servers()) == [1, 2, 3]


def test_servers_beyond_max_distance_are_excluded(st):
    assert ids(st.get_servers(max_distance=500)) == [1, 2]
    assert all(d <= 500 for d in st.servers)


def test_server_exactly_at_max_distance_is_kept(st):
    assert ids(st.get_servers(max_distance=D_B)) == [1, 2]


def test_server_just_beyond_max_distance_is_dropped(st):
    assert ids(st.get_servers(max_distance=D_B - 1e-9)) == [1]


def test_max_distance_excluding_everything_raises(st):
    st.lat_lon = (45.0, 45.0)
    with pytest.raises(speedtest.NoMatchedServers):
        st.get_servers(max_distance=1)


@pytest.mark.parametrize('kwargs', [
    {'max_distance': 500}, {'exclude': [1]}, {'servers': [1]}])
def test_retrieval_failure_with_filter_raises_retrieval_error(
        st, monkeypatch, kwargs):
    monkeypatch.setattr(speedtest, 'catch_request',
                        lambda request, opener=None: (None, 'boom'))
    with pytest.raises(speedtest.ServersRetrievalError):
        st.get_servers(**kwargs)


def test_retrieval_failure_without_filter_returns_empty(st, monkeypatch):
    monkeypatch.setattr(speedtest, 'catch_request',
                        lambda request, opener=None: (None, 'boom'))
    assert st.get_servers() == {}


def test_max_distance_combines_with_server_and_exclude(st):
    assert ids(st.get_servers(exclude=[1], max_distance=D_B)) == [2]
    assert ids(st.get_servers(servers=[2, 3], max_distance=D_B)) == [2]


# ------------------------------------------------------------ shell() wiring

class FakeSpeedtest(object):
    calls = []
    servers = {}
    config = {'client': {'isp': 'ISP', 'ip': '127.0.0.1'}}

    def __init__(self, **kwargs):
        pass

    def get_servers(self, **kwargs):
        FakeSpeedtest.calls.append(kwargs)
        if (kwargs.get('max_distance') is not None or kwargs.get('servers')
                or kwargs.get('exclude')):
            raise speedtest.NoMatchedServers()
        return {}


@pytest.fixture
def run_shell(monkeypatch):
    monkeypatch.setattr(speedtest.signal, 'signal', lambda *a, **k: None)
    monkeypatch.setattr(speedtest, 'Speedtest', FakeSpeedtest)
    FakeSpeedtest.calls = []

    def run(*argv):
        monkeypatch.setattr(sys, 'argv', ['speedtest.py'] + list(argv))
        speedtest.shell()
    return run


@pytest.mark.parametrize('value', ['0', '-5', 'nan'])
def test_shell_rejects_non_positive_max_distance(run_shell, value):
    with pytest.raises(speedtest.SpeedtestCLIError) as exc:
        run_shell('--max-distance', value)
    assert str(exc.value) == '--max-distance must be a positive number'
    assert FakeSpeedtest.calls == []


def test_shell_main_path_no_server_in_range(run_shell):
    with pytest.raises(speedtest.SpeedtestCLIError) as exc:
        run_shell('--simple', '--max-distance', '5')
    assert str(exc.value) == 'No server is within specified range'
    assert FakeSpeedtest.calls == [
        {'servers': None, 'exclude': None, 'max_distance': 5.0}]


@pytest.mark.parametrize('argv, message', [
    (['--server', '42'], 'No matched servers: 42'),
    (['--server', '42', '--server', '43', '--exclude', '1',
      '--max-distance', '5'], 'No matched servers: 42, 43'),
    (['--max-distance', '5'], 'No server is within specified range'),
    (['--exclude', '1', '--max-distance', '5'],
     'No server is within specified range after applying --exclude'),
    (['--exclude', '1'], 'No servers remain after applying --exclude'),
])
def test_shell_main_path_no_matched_messages(run_shell, argv, message):
    with pytest.raises(speedtest.SpeedtestCLIError) as exc:
        run_shell('--simple', *argv)
    assert str(exc.value) == message


def test_shell_list_path_passes_max_distance(run_shell):
    with pytest.raises(speedtest.SpeedtestCLIError) as exc:
        run_shell('--list', '--max-distance', '5')
    assert str(exc.value) == 'No server is within specified range'
    assert FakeSpeedtest.calls == [{'max_distance': 5.0}]


def test_shell_list_path_without_flag_passes_none(run_shell):
    with pytest.raises(SystemExit):
        run_shell('--list')
    assert FakeSpeedtest.calls == [{'max_distance': None}]
