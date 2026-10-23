"""Collect immutable short-run evidence; never loads data/model tensors."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(root, output):
    output.mkdir(parents=True, exist_ok=False)
    runs, profiles = {}, {}
    for path in sorted(root.glob("*/result.json")):
        name = path.parent.name
        result = json.loads(path.read_text())
        manifest = json.loads((path.parent/"manifest.json").read_text())
        runs[name] = dict(result=result, manifest=manifest)
        if name.startswith("profile_v2_"):
            runs[name]["excluded_from_conclusions"] = "superseded: ContextVar worker-thread peak aggregation bug; fixed and rerun as v3"
        if not name.startswith("profile_v4_"):
            continue
        modules = defaultdict(lambda: dict(calls=0, seconds=0., allocated=0, reserved=0,
                                          window_peak_allocated=0, window_peak_reserved=0))
        failures, peak_event, peak, entries = [], None, -1, {}
        with (path.parent/"events.jsonl").open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["event"] == "module_enter":
                    entries[tuple(event["active"])] = event
                if event["event"] in ("module_exit", "module_error"):
                    key = (event["phase"], event["module"])
                    record = modules[key]
                    record["calls"] += 1
                    record["seconds"] += event["seconds"]
                    for field in ("allocated", "reserved", "window_peak_allocated", "window_peak_reserved"):
                        record[field] = max(record[field], event[field])
                    entry = entries.pop(tuple(event["active"]), {})
                    record["entry_allocated"] = max(record.get("entry_allocated",0), entry.get("allocated",0))
                    record["entry_reserved"] = max(record.get("entry_reserved",0), entry.get("reserved",0))
                if event["event"] == "operator_error":
                    failures.append(event)
                if event["event"] == "operator" and event["after"]["peak_allocated"] > peak:
                    peak, peak_event = event["after"]["peak_allocated"], event
        with (output/(name+"_modules.csv")).open("w") as stream:
            fields = ["phase", "module", "calls", "seconds", "entry_allocated", "entry_reserved",
                      "allocated", "reserved", "window_peak_allocated", "window_peak_reserved"]
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
            for (phase,module), record in sorted(modules.items()):
                writer.writerow(dict(phase=phase, module=module, **record))
        profiles[name] = dict(result=result, peak_operator=peak_event, failed_operators=failures,
                              note="module window peaks include live tensors from other modules; times are inclusive")
        if (path.parent/"exception.txt").exists():
            (output/(name+"_exception.txt")).write_bytes((path.parent/"exception.txt").read_bytes())
    (output/"runs.json").write_text(json.dumps(runs, indent=2)+"\n")
    (output/"profiles.json").write_text(json.dumps(profiles, indent=2)+"\n")
    comparisons = {}
    for path in sorted(root.glob("comparison_*.json")):
        data = json.loads(path.read_text())
        comparisons[path.stem] = {key:value for key,value in data.items() if key not in ("gradient_errors", "buffer_errors")}
        (output/path.name).write_bytes(path.read_bytes())
    (output/"comparisons.json").write_text(json.dumps(comparisons, indent=2)+"\n")
    inventory = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "/sources/" in str(path) or path.suffix == ".pth":
            continue
        inventory.append(dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest(path)))
    (output/"artifact_inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")
    for filename in ("unit_tests_final.log",):
        (output/filename).write_bytes((root/filename).read_bytes())
    logs = output/"logs"
    logs.mkdir()
    for path in sorted(root.glob("*.log")):
        if path.name.startswith(("ab_", "timing_", "profile_v4_", "final_raw_", "release_")):
            (logs/path.name).write_bytes(path.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.output)
