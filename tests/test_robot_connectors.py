import pytest

from sentinel_evc import robot_connectors as robots


class FakeSocket:
    def __init__(self, payload):
        self.payload = bytearray(payload)
        self.sent = []
        self.timeout = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def settimeout(self, timeout):
        self.timeout = timeout

    def recv(self, amount):
        result = bytes(self.payload[:amount])
        del self.payload[:amount]
        return result

    def sendall(self, value):
        self.sent.append(value)


def test_mock_profile_is_an_honest_nonhardware_protocol_demo(tmp_path):
    registry = robots.RobotRegistry(tmp_path)
    profile = registry.create("desk demo", "mock")
    result = registry.diagnose(profile["id"])
    assert result["ok"] is True
    assert result["hardware_connected"] is False
    assert result["commands_sent"] == []
    assert profile["hardware_motion"] is False
    assert profile["credentials_stored"] is False
    assert registry.get(profile["id"])["last_diagnostic"]["status"] == "demo_ready"


def test_ur_dashboard_diagnostic_sends_only_official_read_only_queries(tmp_path, monkeypatch):
    payload = (
        b"Connected: Universal Robots Dashboard Server\n"
        b"UR5e\n"
        b"Robotmode: RUNNING\n"
        b"Safetystatus: NORMAL\n"
    )
    connection = FakeSocket(payload)
    monkeypatch.setattr(robots.socket, "create_connection", lambda address, timeout: connection)
    registry = robots.RobotRegistry(tmp_path)
    profile = registry.create("cell one", "ur_dashboard_readonly", "192.0.2.10", 29999)
    result = registry.diagnose(profile["id"], timeout=0.5)
    assert result["status"] == "read_only_connected"
    assert result["readings"] == {"model": "UR5e", "robot_mode": "RUNNING", "safety_status": "NORMAL"}
    assert connection.sent == [b"get robot model\n", b"robotmode\n", b"safetystatus\n"]
    assert result["commands_sent"] == ["get robot model", "robotmode", "safetystatus"]


def test_malformed_dashboard_response_is_a_failed_diagnostic_not_connection_claim(tmp_path, monkeypatch):
    connection = FakeSocket(
        b"Connected: Universal Robots Dashboard Server\nUR5e\nRobotmode: RUNNING\nSafetystatus: UNKNOWN\n"
    )
    monkeypatch.setattr(robots.socket, "create_connection", lambda address, timeout: connection)
    registry = robots.RobotRegistry(tmp_path)
    profile = registry.create("cell", "ur_dashboard_readonly", "robot.local")
    result = registry.diagnose(profile["id"])
    assert result["ok"] is False
    assert result["hardware_connected"] is False
    assert result["error"]["code"] == "READ_ONLY_DIAGNOSTIC_FAILED"
    assert result["commands_sent"] == ["get robot model", "robotmode", "safetystatus"]


def test_profiles_refuse_credentials_unsafe_endpoints_and_traversal(tmp_path):
    registry = robots.RobotRegistry(tmp_path)
    with pytest.raises(ValueError):
        registry.create("bad", "ur_dashboard_readonly", "http://robot", 29999)
    with pytest.raises(ValueError):
        registry.create("bad", "ur_dashboard_readonly", "robot.local", True)
    with pytest.raises(ValueError):
        registry.create("bad", "mock", "robot.local", 29999)
    with pytest.raises(ValueError):
        registry.create("bad", ["mock"])
    with pytest.raises(TypeError):
        registry.create("bad", "ur_dashboard_readonly", "robot.local", password="secret")
    with pytest.raises(KeyError):
        registry.get("../robot-secret")
    linked = tmp_path / "linked-robots"
    linked.symlink_to(tmp_path / "actual-robots", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        robots.RobotRegistry(linked)


def test_persisted_unhashable_driver_is_ignored_instead_of_crashing_list(tmp_path):
    registry = robots.RobotRegistry(tmp_path)
    profile = registry.create("demo", "mock")
    path = tmp_path / profile["id"] / "robot.json"
    import json
    record = json.loads(path.read_text())
    record["driver"] = ["mock"]
    path.write_text(json.dumps(record))
    with pytest.raises(KeyError):
        registry.get(profile["id"])
    assert registry.list() == []


def test_dashboard_line_and_timeout_are_bounded(tmp_path, monkeypatch):
    connection = FakeSocket(b"x" * (robots.MAX_LINE + 1))
    monkeypatch.setattr(robots.socket, "create_connection", lambda address, timeout: connection)
    registry = robots.RobotRegistry(tmp_path)
    profile = registry.create("cell", "ur_dashboard_readonly", "robot.local")
    result = registry.diagnose(profile["id"], timeout=0.1)
    assert result["ok"] is False
    with pytest.raises(ValueError):
        registry.diagnose(profile["id"], timeout=float("inf"))
