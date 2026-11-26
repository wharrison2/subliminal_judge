"""RunPod REST v2 client: create body shape, queued pods, teardown confirmation, billing.

No network: HTTP is replaced by fakes."""
import io
import json
import sys
import types
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pod"))
import pod  # noqa: E402

# The accepted shape of POST /v2/pods, transcribed from the provider's v2 specification
# (CreatePodRequest with its container, GPU and mount parts). Unknown keys are rejected.
CREATE_SCHEMA = {
    "required": ["name", "image"],
    "properties": {
        "name": str, "image": str, "cloud": ("SECURE", "COMMUNITY"), "disk": int,
        "env": dict, "ports": list, "dataCenterIds": list, "startSsh": bool,
        "startJupyter": bool, "globalNetworking": bool, "templateId": str,
        "args": str, "cmd": list, "entrypoint": list, "registry": str,
        "gpu": dict, "cpu": dict, "mounts": dict, "queue": dict,
    },
    "gpu": {"required": ["id"], "allowed": {"id", "count", "allowedCudaVersions", "minCudaVersion",
                                            "minRamPerGpu", "minVcpuCountPerGpu"}},
    "mounts": {"allowed": {"persistent", "network", "global"}},
    "network_mount": {"required": ["volumeId", "path"], "allowed": {"volumeId", "path"}},
}


def check_against_schema(body):
    allowed = CREATE_SCHEMA["properties"]
    assert set(body) <= set(allowed), f"unknown keys {set(body) - set(allowed)}"
    for key in CREATE_SCHEMA["required"]:
        assert key in body
    for key, rule in allowed.items():
        if key not in body:
            continue
        if isinstance(rule, tuple):
            assert body[key] in rule
        else:
            assert isinstance(body[key], rule) and not (rule is int and isinstance(body[key], bool))
    assert exactly_one(body, "gpu", "cpu")
    gpu = body["gpu"]
    assert set(CREATE_SCHEMA["gpu"]["required"]) <= set(gpu) <= CREATE_SCHEMA["gpu"]["allowed"]
    assert isinstance(gpu["id"], str) and gpu.get("count", 1) >= 1
    mounts = body.get("mounts", {})
    assert set(mounts) <= CREATE_SCHEMA["mounts"]["allowed"]
    assert not ("persistent" in mounts and ("network" in mounts or "global" in mounts))
    for mount in mounts.get("network", []):
        assert set(CREATE_SCHEMA["network_mount"]["required"]) <= set(mount) <= CREATE_SCHEMA["network_mount"]["allowed"]


def exactly_one(body, *keys):
    return sum(k in body for k in keys) == 1


def make_args(**overrides):
    args = dict(name="test", image=pod.DEFAULT_IMAGE, gpu_count=2, disk=150, secure=True,
                volume=None, datacenter="US-KS-2", gpu="NVIDIA A100-SXM4-80GB", min_vram=80,
                allow_amd=False, dry_run=False)
    args.update(overrides)
    return types.SimpleNamespace(**args)


def test_create_body_matches_the_spec_and_keeps_our_requirements():
    body = pod.build_create_body(make_args(), "US-KS-2", "NVIDIA A100-SXM4-80GB")
    check_against_schema(body)
    assert body["name"] == "test" and body["image"] == pod.DEFAULT_IMAGE
    assert body["gpu"] == {"id": "NVIDIA A100-SXM4-80GB", "count": 2}
    assert body["cloud"] == "SECURE" and body["disk"] == 150
    assert body["ports"] == ["22/tcp"] and body["env"] == {"HF_HOME": "/root/hf"}
    assert body["dataCenterIds"] == ["US-KS-2"] and body["startSsh"] is True
    assert "mounts" not in body


def test_create_body_without_secure_is_community_and_default_disk_is_150():
    args = pod.build_parser().parse_args(["create", "--name", "x"])
    body = pod.build_create_body(args, args.datacenter, "g")
    check_against_schema(body)
    assert body["cloud"] == "COMMUNITY" and body["disk"] == 150


def test_create_body_with_volume_mounts_workspace_and_forces_secure():
    body = pod.build_create_body(make_args(volume="vol1", secure=False), "US-KS-2", "g")
    check_against_schema(body)
    assert body["mounts"] == {"network": [{"volumeId": "vol1", "path": "/workspace"}]}
    assert body["cloud"] == "SECURE" and "dataCenterIds" not in body


def test_dry_run_makes_no_request(monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(pod.urllib.request, "urlopen", boom)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    pod.main(["create", "--name", "test", "--gpu-count", "2", "--secure",
              "--gpu", "NVIDIA A100-SXM4-80GB", "--datacenter", "US-KS-2", "--dry-run"])
    body = json.loads(capsys.readouterr().out)
    check_against_schema(body)


class FakeApi:
    """Replaces pod.api_with_status with scripted responses keyed by (method, path)."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, method, path, body=None, query=None):
        self.calls.append((method, path, body))
        result = self.routes[(method, path)]
        if isinstance(result, list):
            result = result.pop(0)
        if isinstance(result, pod.PodApiError):
            raise result
        return result


def install(monkeypatch, routes):
    fake = FakeApi(routes)
    monkeypatch.setattr(pod, "api_with_status", fake)
    return fake


def http_error(status):
    return pod.PodApiError(f"x -> {status}: {{}}", status=status)


QUEUED = {"id": "q1", "status": "QUEUED", "podId": None, "name": "test",
          "queue": {"expiresAt": "2027-01-01T00:00:00Z"}}


def test_create_queued_response_is_reported_loudly_with_the_queued_id(monkeypatch, capsys):
    install(monkeypatch, {("POST", "/pods"): (202, QUEUED)})
    pod.command_create(make_args())
    out = capsys.readouterr().out
    assert "NOT RUNNING" in out and "QUEUED_POD_ID=q1" in out and "cancel-queued q1" in out
    assert "POD_ID=" not in out.replace("QUEUED_POD_ID=", "")


def test_create_running_response_prints_pod_id(monkeypatch, capsys):
    install(monkeypatch, {("POST", "/pods"): (201, {"id": "p1", "status": "PROVISIONING", "cost": 3.0})})
    pod.command_create(make_args())
    assert "POD_ID=p1" in capsys.readouterr().out


def test_create_tries_next_gpu_on_400_but_stops_on_server_error(monkeypatch):
    fake = install(monkeypatch, {("POST", "/pods"): [http_error(400), (201, {"id": "p2"})]})
    status, body, gpu = pod.create_pod(make_args(), "dc", ["a", "b"])
    assert (status, gpu) == (201, "b") and [c[2]["gpu"]["id"] for c in fake.calls] == ["a", "b"]
    fake = install(monkeypatch, {("POST", "/pods"): [http_error(500), (201, {"id": "p3"})]})
    with pytest.raises(pod.PodApiError):
        pod.create_pod(make_args(), "dc", ["a", "b"])
    assert len(fake.calls) == 1          # no blind retry: the pod might have been created


def test_list_shows_pods_and_queued_pods(monkeypatch, capsys):
    install(monkeypatch, {
        ("GET", "/pods"): (200, {"pods": [{"id": "p1", "status": "RUNNING", "name": "n", "cost": 2.5}],
                                 "pagination": {"nextCursor": None, "hasNextPage": False}}),
        ("GET", "/queued-pods"): (200, {"queuedPods": [QUEUED, {**QUEUED, "id": "q2", "status": "EXPIRED"}],
                                        "pagination": {"nextCursor": None, "hasNextPage": False}})})
    pod.command_list(None)
    out = capsys.readouterr().out
    assert "p1" in out and "q1" in out and "q2" not in out


def test_list_follows_pagination(monkeypatch):
    install(monkeypatch, {("GET", "/pods"): [
        (200, {"pods": [{"id": "a"}], "pagination": {"nextCursor": "c", "hasNextPage": True}}),
        (200, {"pods": [{"id": "b"}], "pagination": {"nextCursor": None, "hasNextPage": False}})]})
    assert [p["id"] for p in pod.api_list("/pods", "pods")] == ["a", "b"]


def test_status_and_terminate_handle_a_queued_id(monkeypatch, capsys):
    fake = install(monkeypatch, {
        ("GET", "/pods/q1"): http_error(404), ("GET", "/queued-pods/q1"): (200, QUEUED),
        ("DELETE", "/queued-pods/q1"): (204, {})})
    pod.command_status(types.SimpleNamespace(pod_id="q1"))
    assert "QUEUED_POD_ID=q1" in capsys.readouterr().out
    pod.terminate_any("q1")
    assert ("DELETE", "/queued-pods/q1", None) in fake.calls
    assert not any(c[:2] == ("DELETE", "/pods/q1") for c in fake.calls)


def test_terminate_deletes_the_pod(monkeypatch):
    fake = install(monkeypatch, {("GET", "/pods/p1"): (200, {"id": "p1", "cost": 2.0, "status": "RUNNING"}),
                                 ("DELETE", "/pods/p1"): (204, {})})
    pod.terminate_any("p1")
    assert ("DELETE", "/pods/p1", None) in fake.calls


def test_stop_and_start_use_the_action_endpoint(monkeypatch):
    fake = install(monkeypatch, {("POST", "/pods/p1/action"): (200, {})})
    pod.command_action("stop")(types.SimpleNamespace(pod_id="p1"))
    pod.command_action("start")(types.SimpleNamespace(pod_id="p1"))
    assert [c[2] for c in fake.calls] == [{"action": "stop"}, {"action": "start"}]


EMPTY_LIST = (200, {"pods": [], "pagination": {"nextCursor": None, "hasNextPage": False}})


def test_teardown_confirmed_on_404_with_empty_list(monkeypatch):
    install(monkeypatch, {("GET", "/pods/p1"): http_error(404), ("GET", "/pods"): EMPTY_LIST})
    assert pod.pod_is_gone("p1") is True


def test_teardown_confirmed_on_terminated_status(monkeypatch):
    install(monkeypatch, {("GET", "/pods/p1"): (200, {"id": "p1", "status": "TERMINATED"}),
                          ("GET", "/pods"): EMPTY_LIST})
    assert pod.pod_is_gone("p1") is True


@pytest.mark.parametrize("answer", [
    (200, {"id": "p1", "status": "RUNNING"}), (200, {"id": "p1", "status": "EXITED"}),
    http_error(401), http_error(403), http_error(500), http_error(429),
    pod.PodApiError("no response")])
def test_teardown_not_confirmed_unless_truly_gone(monkeypatch, answer):
    install(monkeypatch, {("GET", "/pods/p1"): answer, ("GET", "/pods"): EMPTY_LIST})
    assert pod.pod_is_gone("p1") is False


def test_teardown_not_confirmed_if_still_listed_or_list_fails(monkeypatch):
    listed = (200, {"pods": [{"id": "p1", "status": "RUNNING"}],
                    "pagination": {"nextCursor": None, "hasNextPage": False}})
    install(monkeypatch, {("GET", "/pods/p1"): http_error(404), ("GET", "/pods"): listed})
    assert pod.pod_is_gone("p1") is False
    install(monkeypatch, {("GET", "/pods/p1"): http_error(404), ("GET", "/pods"): http_error(401)})
    assert pod.pod_is_gone("p1") is False


def test_watcher_uses_the_same_gone_check(monkeypatch):
    import watch_and_tear_down as watch
    install(monkeypatch, {("GET", "/pods/p1"): (200, {"status": "RUNNING"})})
    ops = watch.RealOperations.__new__(watch.RealOperations)
    assert ops.pod_is_gone("p1") is False


def test_spend_and_ssh_helpers_read_v2_field_names():
    import datetime as dt
    now = dt.datetime(2026, 12, 1, 12, 0, tzinfo=dt.timezone.utc)
    rate, hours, spend = pod.estimate_spend(
        {"cost": 8.0, "startedAt": "2026-12-01T10:00:00Z", "status": "RUNNING"}, now)
    assert (rate, round(hours, 2), round(spend, 2)) == (8.0, 2.0, 16.0)
    # a stopped pod reports a rate but is not accruing compute cost
    assert pod.estimate_spend({"cost": 8.0, "startedAt": "2026-12-01T10:00:00Z", "status": "EXITED"}, now) == (8.0, None, None)
    direct = {"ssh": {"proxy": None, "direct": {"host": "1.1.1.1", "port": 4000}}}
    assert pod.ssh_endpoint(direct) == ("1.1.1.1", 4000)
    runtime = {"ssh": {"direct": None}, "runtime": {"ports": [{"private": 22, "public": 4001, "ip": "2.2.2.2"}]}}
    assert pod.ssh_endpoint(runtime) == ("2.2.2.2", 4001)
    assert pod.ssh_endpoint({"ssh": {"direct": None}}) == (None, None)


def test_spend_prints_billed_totals(monkeypatch, capsys):
    record = {"podId": "p1", "startTime": "2026-12-01T00:00:00Z", "totalAmount": 12.5,
              "gpuAmount": 12.0, "cpuAmount": 0.0, "diskAmount": 0.5}
    install(monkeypatch, {
        ("GET", "/billing/pods"): (200, {"records": [record, {**record, "totalAmount": 0}],
                                          "metadata": {"totals": {"totalAmount": 12.5}}}),
        ("GET", "/billing/network-volumes"): (200, {"records": [
            {"networkVolumeId": "v", "startTime": "2026-12-01T00:00:00Z", "totalAmount": 1.0}]})})
    pod.command_spend(types.SimpleNamespace(days=30, pod=None))
    out = capsys.readouterr().out
    assert "total billed: $13.50" in out and "1 billed records" in out


def test_http_layer_sends_v2_url_user_agent_and_status_attribute(monkeypatch):
    seen = {}

    class Response(io.BytesIO):
        status = 201
        __enter__ = lambda self: self
        __exit__ = lambda self, *a: False

    def fake_urlopen(request, timeout=None):
        seen.update(url=request.full_url, method=request.get_method(),
                    agent=request.get_header("User-agent"))
        return Response(b'{"id": "p1"}')
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(pod.urllib.request, "urlopen", fake_urlopen)
    assert pod.api_with_status("POST", "/pods", {"a": 1}) == (201, {"id": "p1"})
    assert seen["url"] == "https://api.runpod.io/v2/pods" and seen["agent"].startswith("curl/")

    def not_found(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "nf", {}, io.BytesIO(b"{}"))
    monkeypatch.setattr(pod.urllib.request, "urlopen", not_found)
    with pytest.raises(pod.PodApiError) as caught:
        pod.api("GET", "/pods/x")
    assert caught.value.status == 404
