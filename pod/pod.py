#!/usr/bin/env python3
"""Create, inspect, stop and terminate RunPod pods (REST API), and report spend.

    python pod.py gpus   [--datacenter US-KS-2] [--min-vram 80] [--count 4]
    python pod.py create --name <descriptive_pod_name> --gpu-count 4 [--volume <id>]
    python pod.py list
    python pod.py status <pod_id>          # includes hourly rate and estimated spend so far
    python pod.py ssh    <pod_id>          # prints the ssh command and the POD_* variables
    python pod.py stop | start | terminate <pod_id>
    python pod.py spend                    # billing records, if the API provides them

Pods bill wall-clock time while they exist, not while the GPU is busy. Always terminate
a pod you are done with (after copying the results off), and confirm with `status`.

Credentials: the RunPod API key is read only from the environment variable RUNPOD_API_KEY.
It is never accepted as a command-line argument and never printed.
"""
import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request

REST_API = "https://rest.runpod.io/v1"
GRAPHQL_API = "https://api.runpod.io/graphql"
# Cloudflare rejects the default Python-urllib User-Agent.
USER_AGENT = "curl/8.4.0"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


class PodApiError(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise PodApiError(
            "RUNPOD_API_KEY is not set. Export it in your shell (do not pass it as an "
            "argument, and never commit it).")
    return key


def api(method: str, path: str, body=None):
    request = urllib.request.Request(
        f"{REST_API}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {api_key()}",
                 "Content-Type": "application/json", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        raise PodApiError(f"{method} {path} -> {error.code}: {error.read().decode()[:800]}")


def as_list(value):
    return value if isinstance(value, list) else (value or {}).get("data", [])


def gpu_stock(datacenter=None, min_vram=0, count=1, allow_amd=False):
    """GPU types with live stock and a price, cheapest first.

    GPU metadata is only on the GraphQL API (REST has no GPU-types endpoint). Passing a
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

    This is an estimate from the hourly rate and the last start time. For recorded spend
    use `spend` (billing records) or the provider's billing page.
    """
    rate = pod.get("costPerHr") or pod.get("adjustedCostPerHr")
    started = pod.get("lastStartedAt")
    if rate is None or not started:
        return rate, None, None
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
    """(ip, public_port) for SSH, or (None, None) if the pod has no mapping yet."""
    port_mappings = pod.get("portMappings") or {}
    if isinstance(port_mappings, dict):
        port = port_mappings.get("22")
    else:
        port = next((m.get("publicPort") for m in port_mappings
                     if str(m.get("privatePort")) == "22"), None)
    return pod.get("publicIp"), port


def build_create_body(args, datacenter, gpu_ids):
    body = {
        "name": args.name, "imageName": args.image, "gpuCount": args.gpu_count,
        "gpuTypeIds": gpu_ids, "gpuTypePriority": "availability",
        "containerDiskInGb": args.disk,
        "cloudType": "SECURE" if (args.secure or args.volume) else "COMMUNITY",
        "ports": ["22/tcp"],
        # Model weights go on the container disk; only small outputs belong on /workspace.
        "env": {"HF_HOME": "/root/hf"},
    }
    if args.volume:
        body["networkVolumeId"] = args.volume
        body["volumeMountPath"] = "/workspace"
    else:
        body["dataCenterIds"] = [datacenter]
        body["dataCenterPriority"] = "availability"
    return body


def print_spend(pod: dict):
    rate, hours, spend = estimate_spend(pod)
    if rate is None:
        print("  cost: provider did not report an hourly rate")
        return
    line = f"  cost: ${rate:.2f}/hr (whole pod)"
    if spend is not None:
        line += f"; running {hours:.2f} h since last start; estimated ${spend:.2f} this session"
    print(line)


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
        volumes = as_list(api("GET", "/networkvolumes"))
        match = next((v for v in volumes if v.get("id") == a.volume), None)
        if not match:
            raise PodApiError(f"Volume {a.volume} not found (use vol-list).")
        datacenter = match.get("dataCenterId")  # a volume pins the datacenter
    if a.gpu:
        gpu_ids = [a.gpu]
    else:
        stock = gpu_stock(datacenter, a.min_vram, a.gpu_count, a.allow_amd)
        if not stock:
            raise PodApiError(f"No GPU >= {a.min_vram} GB with stock in {datacenter}.")
        gpu_ids = [g["id"] for g in stock[:12]]
    body = build_create_body(a, datacenter, gpu_ids)
    if a.dry_run:
        print(json.dumps(body, indent=1))
        return
    pod = api("POST", "/pods", body)
    pod_id = pod.get("id") or pod.get("podId")
    print(f"created pod {a.name!r}\nPOD_ID={pod_id}")
    print_spend(pod)
    print(f"billing runs until you run:  python pod.py terminate {pod_id}")


def command_list(a):
    pods = as_list(api("GET", "/pods"))
    if not pods:
        print("(no pods)")
    for p in pods:
        rate = p.get("costPerHr")
        print(f"  {p.get('id'):16} {p.get('desiredStatus'):10} "
              f"{p.get('name')}  ${rate}/hr" if rate is not None else
              f"  {p.get('id'):16} {p.get('desiredStatus'):10} {p.get('name')}")


def command_status(a):
    pod = api("GET", f"/pods/{a.pod_id}")
    keep = ("id", "name", "desiredStatus", "gpu", "image", "publicIp", "portMappings",
            "costPerHr", "lastStartedAt", "containerDiskInGb", "volumeInGb")
    print(json.dumps({k: pod.get(k) for k in keep if k in pod}, indent=1))
    print_spend(pod)


def command_ssh(a):
    pod = api("GET", f"/pods/{a.pod_id}")
    ip, port = ssh_endpoint(pod)
    if not (ip and port):
        raise PodApiError(f"No public SSH mapping yet (status={pod.get('desiredStatus')}).")
    key = a.key
    print(f"ssh root@{ip} -p {port} -i {key}")
    print("\n# for sync.py and jobs.py:")
    print(f"export POD_HOST={ip} POD_PORT={port} POD_SSH_KEY={key}")


def command_action(action):
    def run(a):
        if action == "terminate":
            try:
                print_spend(api("GET", f"/pods/{a.pod_id}"))  # last look at the rate
            except PodApiError:
                pass
            api("DELETE", f"/pods/{a.pod_id}")
            print(f"terminated {a.pod_id}; confirm with: python pod.py status {a.pod_id} "
                  "(expect a 404)")
        else:
            api("POST", f"/pods/{a.pod_id}/{action}")
            print(f"{action} requested for {a.pod_id}")
    return run


def command_spend(a):
    try:
        records = as_list(api("GET", "/billing/pods"))
    except PodApiError as error:
        raise PodApiError(f"Billing records unavailable ({error}). Use the provider's "
                          "billing page and write the figure down.")
    total = 0.0
    for r in records:
        amount = r.get("amount") or 0
        total += amount
        print(f"  {r.get('time', '?')}  pod {r.get('podId', '?')}  ${amount:.2f}")
    print(f"total over {len(records)} records: ${total:.2f}")


def command_volumes(a):
    volumes = as_list(api("GET", "/networkvolumes"))
    if not volumes:
        print("(no network volumes)")
    for v in volumes:
        print(f"  {v.get('id'):24} {v.get('name')}  {v.get('size')} GB  {v.get('dataCenterId')}")


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
    sub.add_parser("spend", help="billing records").set_defaults(fn=command_spend)
    for name, fn in (("status", command_status), ("ssh", command_ssh),
                     ("stop", command_action("stop")), ("start", command_action("start")),
                     ("terminate", command_action("terminate"))):
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
