#!/usr/bin/env python3
"""Create, inspect, stop and terminate RunPod pods (REST API v2), and report spend.

    python pod.py gpus   [--datacenter US-KS-2] [--min-vram 80] [--count 4]
    python pod.py create --name <descriptive_pod_name> --gpu-count 4 [--volume <id>]
    python pod.py list
    python pod.py status <pod_id>          # includes hourly rate and estimated spend so far
    python pod.py ssh    <pod_id>          # prints the ssh command and the POD_* variables
    python pod.py stop | start | terminate <pod_id>
    python pod.py cancel-queued <queued_pod_id>
    python pod.py volumes
    python pod.py spend [--days 30]        # billed amounts from the billing API

Pods bill wall-clock time while they exist, not while the GPU is busy. Always terminate
a pod you are done with (after copying the results off), and confirm with `status`
(a terminated pod answers 404, or reports status TERMINATED).

A create can in principle return a queued pod (HTTP 202, no pod exists yet) instead of a
running one. This client never asks for queueing, but it handles the response: the queued id
is printed, `list` shows queued pods, and `status`, `terminate` and `cancel-queued` accept it.

Credentials: the RunPod API key is read only from the environment variable RUNPOD_API_KEY.
It is never accepted as a command-line argument and never printed.
"""
import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API_ROOT = "https://api.runpod.io"
REST_API = f"{API_ROOT}/v2"
GRAPHQL_API = f"{API_ROOT}/graphql"
# Cloudflare rejects the default Python-urllib User-Agent.
USER_AGENT = "curl/8.4.0"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
REQUEST_TIMEOUT_SECONDS = 60
QUEUED_STATUSES = {"QUEUED", "DEPLOYED", "FAILED", "EXPIRED", "CANCELLED"}


class PodApiError(RuntimeError):
    """An API call failed. `status` is the HTTP status, or None if no response arrived."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise PodApiError(
            "RUNPOD_API_KEY is not set. Export it in your shell (do not pass it as an "
            "argument, and never commit it).")
    return key


def api_with_status(method: str, path: str, body=None, query=None):
    """(http_status, parsed JSON body or {}) for one call; raises PodApiError otherwise."""
    url = f"{REST_API}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(
        url, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {api_key()}",
                 "Content-Type": "application/json", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw = response.read()
            return response.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as error:
        raise PodApiError(f"{method} {path} -> {error.code}: {error.read().decode()[:800]}",
                          status=error.code)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise PodApiError(f"{method} {path} -> no response ({error}); the request may or "
                          "may not have been applied: check `pod.py list`")


def api(method: str, path: str, body=None, query=None):
    return api_with_status(method, path, body, query)[1]


def api_list(path: str, key: str, query=None):
    """Every item of a cursor-paginated list endpoint (follows pagination.nextCursor)."""
    items, cursor = [], None
    while True:
        page = api("GET", path, query={**(query or {}), **({"cursor": cursor} if cursor else {})})
        items += page.get(key) or []
        pagination = page.get("pagination") or {}
        cursor = pagination.get("nextCursor")
        if not (pagination.get("hasNextPage") and cursor):
            return items


def gpu_stock(datacenter=None, min_vram=0, count=1, allow_amd=False):
    """GPU types with live stock and a price, cheapest first.

    Uses the GraphQL API, which gives the lowest price and a stock status for a given GPU
    count and datacenter. The REST v2 catalog (`/v2/catalog/gpus`) only offers a per-GPU list
    price and a coarse availability level, so it is not a drop-in replacement. Passing a
    datacenter makes the price null for cards that datacenter cannot serve.
    """
    arg = f"gpuCount:{count}" + (f',dataCenterId:"{datacenter}"' if datacenter else "")
    query = ("query{ gpuTypes { id memoryInGb secureCloud "
             f"lowestPrice(input:{{{arg}}}){{ uninterruptablePrice stockStatus }} }} }}")
    request = urllib.request.Request(
        GRAPHQL_API, method="POST", data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        types = (json.load(response).get("data") or {}).get("gpuTypes") or []
    return filter_gpus(types, min_vram, allow_amd)


def filter_gpus(types, min_vram=0, allow_amd=False):
    def price(gpu):
        return (gpu.get("lowestPrice") or {}).get("uninterruptablePrice")
    kept = [g for g in types
            if (g.get("memoryInGb") or 0) >= min_vram and price(g) is not None
            and (allow_amd or "NVIDIA" in g["id"])]
    return sorted(kept, key=price)


def estimate_spend(pod: dict, now=None):
    """(hourly_rate, hours_running, estimated_spend) from a pod record; None if unknown.

    This is an ESTIMATE from the pod's hourly rate (`cost`) and its last start time
    (`startedAt`); it is not a billed amount. For billed amounts use `spend`. The rate is
    reported in every status, so the hours and the estimate are only given while the pod
    is live (provisioning, starting or running).
    """
    rate = pod.get("cost", pod.get("costPerHr"))
    started = pod.get("startedAt") or pod.get("lastStartedAt")
    if rate is None:
        return None, None, None
    if not started or pod.get("status") not in (None, "PROVISIONING", "STARTING", "RUNNING"):
        return float(rate), None, None
    try:
        start = dt.datetime.fromisoformat(started.replace(" +0000 UTC", "+00:00")
                                          .replace("Z", "+00:00").split(" m=")[0])
    except ValueError:
        return float(rate), None, None
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    now = now or dt.datetime.now(dt.timezone.utc)
    hours = max((now - start).total_seconds() / 3600, 0.0)
    return float(rate), hours, float(rate) * hours


def ssh_endpoint(pod: dict):
    """(ip, public_port) for direct SSH, or (None, None) if the pod has no mapping yet.

    Uses the `ssh.direct` block of a single-pod GET. The list endpoint leaves it null, so
    a pod record from `list` has no endpoint. Falls back to the live port list in `runtime`.
    """
    direct = (pod.get("ssh") or {}).get("direct") or {}
    if direct.get("host") and direct.get("port"):
        return direct["host"], direct["port"]
    for port in (pod.get("runtime") or {}).get("ports") or []:
        if port.get("private") == 22 and port.get("public") and port.get("ip"):
            return port["ip"], port["public"]
    return None, None


def build_create_body(args, datacenter, gpu_id):
    """Request body for POST /v2/pods. v2 places exactly one GPU type per request; the
    caller tries candidate GPU types in order (see `create_pod`)."""
    body = {
        "name": args.name, "image": args.image,
        "gpu": {"id": gpu_id, "count": args.gpu_count},
        "disk": args.disk,
        "cloud": "SECURE" if (args.secure or args.volume) else "COMMUNITY",
        "ports": ["22/tcp"],
        # Registered account SSH keys are injected only when this is set.
        "startSsh": True,
        # Model weights go on the container disk; only small outputs belong on /workspace.
        "env": {"HF_HOME": "/root/hf"},
    }
    if args.volume:
        body["mounts"] = {"network": [{"volumeId": args.volume, "path": "/workspace"}]}
    else:
        body["dataCenterIds"] = [datacenter]
    return body


def is_queued_response(status, body):
    """True if a create answered with a queued pod (HTTP 202, or a queued-pod record)."""
    return status == 202 or (isinstance(body, dict) and "queue" in body
                             and body.get("status") in QUEUED_STATUSES)


def create_pod(args, datacenter, gpu_ids):
    """Try each GPU type in order. Returns (http_status, body, gpu_id) of the first accepted
    create. 400 (no capacity, or this combination not placeable) and 403 (pool not
    accessible) move on to the next candidate; anything else stops. A server error or a
    timeout is NOT retried, because the pod may have been created: check `list`."""
    last_error = None
    for gpu_id in gpu_ids:
        try:
            status, body = api_with_status("POST", "/pods", build_create_body(args, datacenter, gpu_id))
            return status, body, gpu_id
        except PodApiError as error:
            if error.status not in (400, 403):
                raise
            print(f"  {gpu_id}: not placed ({error})", file=sys.stderr)
            last_error = error
    raise PodApiError(f"No candidate GPU type could be placed ({len(gpu_ids)} tried). "
                      f"Last error: {last_error}")


def pod_is_gone(pod_id: str) -> bool:
    """True only if the pod is truly gone. All of these must hold:
    - GET /pods/{id} answers 404, or 200 with status TERMINATED (the spec's status for a
      permanently deleted pod);
    - a fresh list of the account's pods succeeded (so the key works and we are looking at
      the right account, not merely at a 404 caused by a wrong key) and has no live pod with
      that id.
    A network failure, any other status code, or a pod still listed gives False."""
    try:
        pod = api("GET", f"/pods/{pod_id}")
        if pod.get("status") != "TERMINATED":
            return False
    except PodApiError as error:
        if error.status != 404:
            return False
    try:
        listed = api_list("/pods", "pods")
    except PodApiError:
        return False
    return not any(p.get("id") == pod_id and p.get("status") != "TERMINATED" for p in listed)


def print_spend(pod: dict):
    rate, hours, spend = estimate_spend(pod)
    if rate is None:
        print("  cost: provider did not report an hourly rate")
        return
    line = f"  cost: ${rate:.2f}/hr (whole pod, rate only)"
    if spend is not None:
        line += f"; running {hours:.2f} h since last start; ESTIMATED ${spend:.2f} this session"
    print(line)


def print_queued(queued: dict):
    queue = queued.get("queue") or {}
    print(f"QUEUED_POD_ID={queued.get('id')}  status={queued.get('status')}  "
          f"name={queued.get('name')!r}")
    print(f"  no pod exists yet; it deploys when capacity appears (waits until "
          f"{queue.get('expiresAt', '?')}). Once deployed, `status` shows its pod id "
          f"(podId={queued.get('podId')}).")
    print(f"  cancel it with:  python pod.py cancel-queued {queued.get('id')}")


def get_queued_pod(queued_id):
    """The queued-pod record, or None if there is no such queued pod."""
    try:
        return api("GET", f"/queued-pods/{queued_id}")
    except PodApiError as error:
        if error.status == 404:
            return None
        raise


def command_gpus(a):
    gpus = gpu_stock(a.datacenter, a.min_vram, a.count, a.allow_amd)
    where = a.datacenter or "anywhere"
    if not gpus:
        print(f"no GPU >= {a.min_vram} GB with stock in {where}")
        return
    print(f"{'id':46}{'VRAM':>6}{'$/hr each':>10}  stock in {where}")
    for g in gpus:
        lp = g["lowestPrice"]
        print(f"  {g['id'][:44]:44}{g.get('memoryInGb'):>6}"
              f"{lp.get('uninterruptablePrice'):>10}  {lp.get('stockStatus')}")


def command_create(a):
    datacenter = a.datacenter
    if a.volume:
        match = next((v for v in api_list("/network-volumes", "networkVolumes")
                      if v.get("id") == a.volume), None)
        if not match:
            raise PodApiError(f"Volume {a.volume} not found (use `volumes`).")
        datacenter = match.get("dataCenter")  # a volume pins the datacenter
    if a.gpu:
        gpu_ids = [a.gpu]
    else:
        stock = gpu_stock(datacenter, a.min_vram, a.gpu_count, a.allow_amd)
        if not stock:
            raise PodApiError(f"No GPU >= {a.min_vram} GB with stock in {datacenter}.")
        gpu_ids = [g["id"] for g in stock[:12]]
    if a.dry_run:
        print(json.dumps(build_create_body(a, datacenter, gpu_ids[0]), indent=1))
        if len(gpu_ids) > 1:
            print(f"# fallback GPU types, tried in order if this one cannot be placed: {gpu_ids[1:]}")
        return
    status, pod, gpu_id = create_pod(a, datacenter, gpu_ids)
    if is_queued_response(status, pod):
        print(f"NOT RUNNING: pod {a.name!r} was QUEUED, not created (HTTP {status}).")
        print_queued(pod)
        return
    pod_id = pod.get("id")
    print(f"created pod {a.name!r} on {gpu_id}\nPOD_ID={pod_id}  status={pod.get('status')}")
    print_spend(pod)
    print(f"billing runs until you run:  python pod.py terminate {pod_id}")


def command_list(a):
    pods = api_list("/pods", "pods")
    try:
        queued = [q for q in api_list("/queued-pods", "queuedPods") if q.get("status") == "QUEUED"]
        queued_note = ""
    except PodApiError as error:
        queued, queued_note = [], f"(could not list queued pods: {error})"
    if not pods and not queued:
        print("(no pods)")
    for p in pods:
        rate = p.get("cost")
        suffix = f"  ${rate}/hr" if rate is not None else ""
        print(f"  {p.get('id'):16} {p.get('status'):12} {p.get('name')}{suffix}")
    for q in queued:
        print(f"  {q.get('id'):16} {'QUEUED':12} {q.get('name')}  (queued pod: no pod exists yet)")
    if queued_note:
        print(queued_note)


def command_status(a):
    try:
        pod = api("GET", f"/pods/{a.pod_id}")
    except PodApiError as error:
        queued = get_queued_pod(a.pod_id) if error.status == 404 else None
        if queued is None:
            raise
        print_queued(queued)
        return
    keep = ("id", "name", "status", "actions", "gpu", "image", "cloud", "dataCenterId", "ssh",
            "ports", "mounts", "cost", "startedAt", "disk")
    print(json.dumps({k: pod.get(k) for k in keep if k in pod}, indent=1))
    print_spend(pod)


def command_ssh(a):
    pod = api("GET", f"/pods/{a.pod_id}")
    ip, port = ssh_endpoint(pod)
    if not (ip and port):
        raise PodApiError(f"No public SSH mapping yet (status={pod.get('status')}).")
    key = a.key
    print(f"ssh root@{ip} -p {port} -i {key}")
    print("\n# for sync.py and jobs.py:")
    print(f"export POD_HOST={ip} POD_PORT={port} POD_SSH_KEY={key}")


def cancel_queued(queued_id):
    api("DELETE", f"/queued-pods/{queued_id}")
    print(f"cancelled queued pod {queued_id}; it will not deploy. Confirm with: "
          f"python pod.py status {queued_id} (status CANCELLED)")


def command_cancel_queued(a):
    cancel_queued(a.pod_id)


def terminate_any(pod_id):
    """Terminate a pod, or cancel it if the id is a queued pod (or terminate the pod a
    queued pod already deployed as)."""
    try:
        print_spend(api("GET", f"/pods/{pod_id}"))  # last look at the rate
    except PodApiError as error:
        if error.status == 404:
            queued = get_queued_pod(pod_id)
            if queued is not None:
                if queued.get("status") == "QUEUED":
                    cancel_queued(pod_id)
                elif queued.get("podId"):
                    print(f"{pod_id} is a queued pod that already deployed as pod "
                          f"{queued['podId']}; terminating that pod.")
                    return terminate_any(queued["podId"])
                else:
                    print(f"queued pod {pod_id} is {queued.get('status')}; nothing is running.")
                return
    api("DELETE", f"/pods/{pod_id}")
    print(f"terminated {pod_id}; confirm with: python pod.py status {pod_id} "
          "(expect a 404, or status TERMINATED)")


def command_action(action):
    def run(a):
        if action == "terminate":
            terminate_any(a.pod_id)
        else:
            api("POST", f"/pods/{a.pod_id}/action", {"action": action})
            print(f"{action} requested for {a.pod_id}")
    return run


def command_spend(a):
    """Billed amounts per pod per day (GET /billing/pods) and per network volume.

    ACTUAL billing: every `totalAmount`/`gpuAmount`/`cpuAmount`/`diskAmount` below is a billed
    USD amount from the billing API (the most recent bucket is still in progress and may
    grow). ESTIMATES: only the hourly-rate figures `status` prints for a live pod.
    """
    query = {"bucketSize": "day", "lastN": a.days}
    if a.pod:
        query["podId"] = a.pod
    try:
        pod_billing = api("GET", "/billing/pods", query=query)
        volume_billing = api("GET", "/billing/network-volumes", query={"bucketSize": "day", "lastN": a.days})
    except PodApiError as error:
        raise PodApiError(f"Billing records unavailable ({error}). Use the provider's "
                          "billing page and write the figure down.")
    records = [r for r in pod_billing.get("records") or [] if r.get("totalAmount")]
    for r in records:
        print(f"  {r.get('startTime', '?')[:10]}  pod {r.get('podId', '?')}  "
              f"${r.get('totalAmount', 0):.2f}  (gpu ${r.get('gpuAmount', 0):.2f}, "
              f"cpu ${r.get('cpuAmount', 0):.2f}, disk ${r.get('diskAmount', 0):.2f})")
    pod_total = sum(r.get("totalAmount", 0) for r in records)
    volume_records = [r for r in volume_billing.get("records") or [] if r.get("totalAmount")]
    for r in volume_records:
        print(f"  {r.get('startTime', '?')[:10]}  network volume {r.get('networkVolumeId', '?')}  "
              f"${r.get('totalAmount', 0):.2f}")
    volume_total = sum(r.get("totalAmount", 0) for r in volume_records)
    print(f"pods, last {a.days} days: {len(records)} billed records, ${pod_total:.2f} "
          f"(API reports ${(pod_billing.get('metadata') or {}).get('totals', {}).get('totalAmount', 0):.2f})")
    print(f"network volumes, last {a.days} days: {len(volume_records)} billed records, ${volume_total:.2f}")
    print(f"total billed: ${pod_total + volume_total:.2f}  (actual billing amounts; "
          "the last day's bucket is still in progress)")


def command_volumes(a):
    volumes = api_list("/network-volumes", "networkVolumes")
    if not volumes:
        print("(no network volumes)")
    for v in volumes:
        print(f"  {v.get('id'):24} {v.get('name')}  {v.get('size')} GB  {v.get('dataCenter')}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="create a pod")
    c.set_defaults(fn=command_create)
    c.add_argument("--name", required=True,
                   help="full descriptive name, e.g. dpo_seed_grid_four_h100_2026_12_01")
    c.add_argument("--gpu-count", type=int, default=1)
    c.add_argument("--volume", default=None, help="network volume id (forces Secure Cloud)")
    c.add_argument("--datacenter", default="US-CA-2")
    c.add_argument("--disk", type=int, default=150, help="container disk, GB")
    c.add_argument("--min-vram", type=int, default=80)
    c.add_argument("--gpu", default=None, help="exact GPU type id; skips discovery")
    c.add_argument("--allow-amd", action="store_true")
    c.add_argument("--secure", action="store_true")
    c.add_argument("--image", default=DEFAULT_IMAGE)
    c.add_argument("--dry-run", action="store_true", help="print the request, create nothing")
    g = sub.add_parser("gpus", help="list GPU types with stock")
    g.set_defaults(fn=command_gpus)
    g.add_argument("--datacenter", default=None)
    g.add_argument("--min-vram", type=int, default=80)
    g.add_argument("--count", type=int, default=1)
    g.add_argument("--allow-amd", action="store_true")
    sub.add_parser("list", help="list pods").set_defaults(fn=command_list)
    sub.add_parser("volumes", help="list network volumes").set_defaults(fn=command_volumes)
    sp = sub.add_parser("spend", help="billed amounts from the billing API")
    sp.set_defaults(fn=command_spend)
    sp.add_argument("--days", type=int, default=30, help="how many days back (default 30)")
    sp.add_argument("--pod", default=None, help="only this pod id")
    for name, fn in (("status", command_status), ("ssh", command_ssh),
                     ("stop", command_action("stop")), ("start", command_action("start")),
                     ("terminate", command_action("terminate")),
                     ("cancel-queued", command_cancel_queued)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("pod_id")
        if name == "ssh":
            s.add_argument("--key", default="~/.ssh/id_ed25519")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except PodApiError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
