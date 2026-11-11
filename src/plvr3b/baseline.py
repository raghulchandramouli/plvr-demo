from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time
import traceback

import yaml

from .baseline_tasks import build_smoke_cases, evaluate, manifest, public_input, validate_oracles


def encoded(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded(value))
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rss_peak():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def artifact_bytes(output):
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def swap_status():
    try:
        process = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=3)
        return process.stdout.strip() if process.returncode == 0 else "unavailable: " + process.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return "unavailable: " + str(error)


def summarize(traces):
    return {
        "planned_examples": len(traces),
        "attempted_examples": sum(trace["model_calls"] > 0 for trace in traces),
        "outcomes": dict(sorted(Counter(trace["status"] for trace in traces).items())),
        "model_calls": sum(trace["model_calls"] for trace in traces),
        "tool_calls": sum(len((trace.get("evaluation") or {}).get("events", [])) for trace in traces),
        "prompt_tokens": sum(trace["prompt_tokens"] for trace in traces),
        "generation_tokens": sum(trace["generation_tokens"] for trace in traces),
        "model_seconds": sum(trace["generation_seconds"] for trace in traces),
        "evaluation_seconds": sum(trace["evaluation_seconds"] for trace in traces),
        "example_seconds": sum(trace["elapsed_seconds"] for trace in traces),
    }


def run_baseline(root, output, limit=30, oracle_only=False, backend_factory=None):
    root, output = Path(root), Path(output)
    if not 1 <= limit <= 30:
        raise ValueError("Smoke limit must be between 1 and 30")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    config = yaml.safe_load((root / "configs/model.yaml").read_text())
    limits = yaml.safe_load((root / "configs/evaluation.yaml").read_text())
    if limits["examples"] != 30 or limits["max_tool_calls"] != 6 or limits["smoke_version"] != "smoke-v1":
        raise ValueError("Unsupported smoke corpus or grammar policy")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root / "runs/.baseline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another baseline process holds the model lock") from error
        output.mkdir(exist_ok=False)
        (output / "traces").mkdir()
        cases = build_smoke_cases()[:limit]
        oracle_report = validate_oracles(build_smoke_cases())
        descriptor = json.loads((root / "data/manifests/smoke-v1.json").read_text())
        complete_hash = hashlib.sha256(encoded(manifest(build_smoke_cases())).encode()).hexdigest()
        if descriptor["corpus_sha256"] != complete_hash:
            raise ValueError("Smoke corpus differs from its frozen manifest descriptor")
        data = manifest(cases)
        write_json(output / "manifest.json", data)
        write_json(output / "oracle_checks.json", oracle_report)
        hashes = {name: sha256(root / name) for name in (
            ".python-version", "pyproject.toml", "uv.lock", "configs/model.yaml", "configs/evaluation.yaml",
            "src/plvr3b/baseline_tasks.py", "src/plvr3b/model.py", "src/plvr3b/baseline.py",
            "scripts/run_baseline.py", "scripts/evaluate.py",
            "data/manifests/smoke-v1.json",
        )}
        index = root / config["path"] / "model.safetensors.index.json"
        if index.is_file():
            hashes["model/model.safetensors.index.json"] = sha256(index)
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        git_status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout
        git_diff = subprocess.run(["git", "diff", "--no-ext-diff", "--binary", "HEAD"], cwd=root, capture_output=True, check=True).stdout
        metadata = {
            "started_at": started_at, "pid": os.getpid(), "config": config, "limits": limits,
            "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "load_average": os.getloadavg(),
            "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions())),
            "git": {"head": git_head, "status": git_status, "tracked_diff_sha256": hashlib.sha256(git_diff).hexdigest()},
            "hashes": hashes, "manifest_sha256": sha256(output / "manifest.json"),
            "free_disk_bytes_before": shutil.disk_usage(output).free,
            "swap_before": swap_status(), "oracle_only": oracle_only,
            "measurement_notes": {
                "rss": "Process high-water RSS; Darwin ru_maxrss bytes. Not added to MLX allocator bytes.",
                "mlx": "Cumulative MLX allocator peak, not a per-example peak or total process footprint.",
                "tokens": "Prompt tokens are submitted input lengths, not confirmed prefill work on failed calls. MLX cumulative generated-token count per call includes terminal EOS. Totals include failures and exclude speculative lookahead computations.",
                "tool_calls": "Completed successful mock-tool executions, including completed calls in later-failed programs.",
                "deadline": "Cooperative prefill/token-boundary checks; a Metal operation can overrun 120 seconds.",
                "disk": "Logical bytes of all run artifact files, including this summary; not filesystem allocated blocks.",
                "wall": "From runner entry through finalization, before the final summary rewrite and lock release.",
                "catalog": "Every example receives the complete catalog of all 18 tools.",
            },
        }
        write_json(output / "metadata.json", metadata)
        traces = [{
            "id": case.id, "split": case.split, "status": "not_run", "public_input": public_input(case),
            "prompt": None, "raw_output": "", "token_ids": [], "stream_events": 0,
            "model_calls": 0, "prompt_tokens": 0, "generation_tokens": 0,
            "generation_seconds": 0.0, "evaluation_seconds": 0.0, "elapsed_seconds": 0.0,
            "finish_reason": None, "evaluation": None, "errors": [], "resources": {},
            "output_allowance": config["decoding"]["max_tokens"],
        } for case in cases]
        for trace in traces:
            write_json(output / "traces" / f"{trace['id']}.json", trace)
        backend = None
        load_seconds = 0.0
        load_count = 0
        status = "complete"
        fatal_error = None

        def check_resources():
            if rss_peak() > limits["process_memory_budget_bytes"]:
                raise MemoryError("Process RSS peak exceeded the experiment budget")
            if backend and backend.peak_memory() > config["runtime"]["mlx_memory_limit_bytes"]:
                raise MemoryError("MLX allocator peak exceeded the experiment budget")
            if artifact_bytes(output) > limits["artifact_budget_bytes"]:
                raise OSError("Run artifact disk budget exceeded")
            if shutil.disk_usage(output).free < limits["minimum_free_disk_bytes"]:
                raise OSError("Free disk space fell below the declared minimum")

        try:
            check_resources()
            if not oracle_only:
                if backend_factory is None:
                    from .model import MLXModel
                    backend_factory = MLXModel
                load_started = time.perf_counter()
                try:
                    backend = backend_factory(root, config)
                finally:
                    load_seconds = time.perf_counter() - load_started
                load_count = backend.metadata["model_load_count"]
                if load_count != 1:
                    raise ValueError("Baseline must load exactly one model")
                metadata["model"] = backend.metadata
                write_json(output / "metadata.json", metadata)
                check_resources()
            for case, trace in zip(cases, traces):
                example_started = time.perf_counter()
                deadline = example_started + limits["example_timeout_seconds"]
                trace["status"] = "running"
                path = output / "traces" / f"{case.id}.json"
                write_json(path, trace)
                generation_started = None
                stream = None
                try:
                    check_resources()
                    if oracle_only:
                        evaluation_started = time.perf_counter()
                        trace["evaluation"] = evaluate(json.dumps(case.oracle), case)
                        trace["evaluation_seconds"] = time.perf_counter() - evaluation_started
                        trace["status"] = "oracle_validated"
                    else:
                        prompt, tokens = backend.prepare(public_input(case))
                        trace["prompt"] = prompt
                        if len(tokens) + trace["output_allowance"] > config["runtime"]["max_sequence_tokens"]:
                            raise ValueError("Rendered prompt and full output allowance exceed the sequence budget")
                        trace["prompt_tokens"] = len(tokens)
                        trace["model_calls"] = 1
                        write_json(path, trace)
                        generation_started = time.perf_counter()
                        stream = backend.generate(tokens, deadline)
                        with (output / "traces" / f"{case.id}.tokens.jsonl").open("w") as events:
                            for event in stream:
                                events.write(json.dumps({**event, "elapsed_seconds": time.perf_counter() - example_started}) + "\n")
                                events.flush()
                                trace["raw_output"] += event["text"]
                                trace["token_ids"].append(int(event["token_id"]))
                                trace["stream_events"] += 1
                                trace["generation_tokens"] = event["generation_tokens"]
                                trace["finish_reason"] = event["finish_reason"]
                                if event["prompt_tokens"] != len(tokens):
                                    raise ValueError("Generation prompt-token count differs from prepared prompt")
                                if event["generation_tokens"] > trace["output_allowance"]:
                                    raise ValueError("Generation exceeded the output-token budget")
                                if time.perf_counter() > deadline:
                                    raise TimeoutError("Cooperative example deadline exceeded")
                                check_resources()
                        stream.close()
                        stream = None
                        trace["generation_seconds"] = time.perf_counter() - generation_started
                        generation_started = None
                        if trace["finish_reason"] not in ("stop", "length"):
                            raise RuntimeError("Generation ended without a terminal response")
                        evaluation_started = time.perf_counter()
                        trace["evaluation"] = evaluate(trace["raw_output"], case)
                        trace["evaluation_seconds"] = time.perf_counter() - evaluation_started
                        trace["status"] = trace["evaluation"]["status"]
                        if time.perf_counter() > deadline:
                            raise TimeoutError("Cooperative example deadline exceeded during evaluation")
                except (Exception, KeyboardInterrupt) as error:
                    trace["status"] = "timeout" if isinstance(error, TimeoutError) else "runtime_failure"
                    trace["errors"].append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
                    if trace["model_calls"]:
                        evaluation_started = time.perf_counter()
                        trace["evaluation"] = evaluate(trace["raw_output"], case)
                        trace["evaluation_seconds"] += time.perf_counter() - evaluation_started
                    if isinstance(error, (MemoryError, KeyboardInterrupt)) or isinstance(error, OSError) and not isinstance(error, TimeoutError):
                        raise
                finally:
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception as error:
                            trace["status"] = "runtime_failure"
                            trace["errors"].append({"type": type(error).__name__, "message": str(error), "phase": "generation_cleanup"})
                    if generation_started is not None:
                        trace["generation_seconds"] = time.perf_counter() - generation_started
                    trace["elapsed_seconds"] = time.perf_counter() - example_started
                    trace["resources"] = {
                        "process_peak_rss_bytes": rss_peak(),
                        "mlx_peak_bytes": backend.peak_memory() if backend else 0,
                        "observations": backend.last_observations if backend else {},
                    }
                    write_json(path, trace)
                print(f"{case.id}: {trace['status']} ({trace['generation_tokens']} generated tokens)", flush=True)
        except (Exception, KeyboardInterrupt) as error:
            status = "aborted"
            fatal_error = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        finally:
            for trace in traces:
                if trace["status"] == "not_run":
                    trace["errors"].append({"type": "RunAborted", "message": "Run stopped before this example", "cause": fatal_error})
                    write_json(output / "traces" / f"{trace['id']}.json", trace)
            summary = {
                **summarize(traces), "status": status, "fatal_error": fatal_error,
                "model_load_count": load_count, "model_load_seconds": load_seconds,
                "process_peak_rss_bytes": rss_peak(), "mlx_peak_bytes": backend.peak_memory() if backend else 0,
                "free_disk_bytes_after": shutil.disk_usage(output).free, "swap_after": swap_status(),
                "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.perf_counter() - started,
                "artifact_bytes": 0,
            }
            for _ in range(5):
                write_json(output / "summary.json", summary)
                measured = artifact_bytes(output)
                if summary["artifact_bytes"] == measured:
                    break
                summary["artifact_bytes"] = measured
            write_json(output / "summary.json", summary)
        return summary


def audit_run(output):
    from .baseline_tasks import SmokeCase

    output = Path(output)
    metadata = json.loads((output / "metadata.json").read_text())
    data = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    if sha256(output / "manifest.json") != metadata["manifest_sha256"]:
        raise ValueError("Manifest hash differs from run metadata")
    traces = []
    for raw_case in data["cases"]:
        case = SmokeCase(**raw_case)
        trace = json.loads((output / "traces" / f"{case.id}.json").read_text())
        traces.append(trace)
        if trace["id"] != case.id:
            raise ValueError("Trace ID differs from manifest")
        if trace["evaluation"] is not None:
            raw = json.dumps(case.oracle) if metadata["oracle_only"] else trace["raw_output"]
            replayed = evaluate(raw, case)
            if replayed != trace["evaluation"]:
                raise ValueError(f"Evaluation replay differs for {case.id}")
            if trace["status"] in ("success", "format_failure", "contract_failure", "semantic_failure") and replayed["status"] != trace["status"]:
                raise ValueError(f"Outcome differs for {case.id}")
        elif trace["model_calls"]:
            raise ValueError(f"Attempted model call has no inspectable evaluation for {case.id}")
        event_path = output / "traces" / f"{case.id}.tokens.jsonl"
        events = [json.loads(line) for line in event_path.read_text().splitlines()] if event_path.exists() else []
        if events:
            if "".join(event["text"] for event in events) != trace["raw_output"]:
                raise ValueError(f"Raw stream differs for {case.id}")
            if [event["token_id"] for event in events] != trace["token_ids"]:
                raise ValueError(f"Token IDs differ for {case.id}")
            if events[-1]["generation_tokens"] != trace["generation_tokens"]:
                raise ValueError(f"Generated-token total differs for {case.id}")
        if trace["model_calls"] and trace["prompt_tokens"] + trace["generation_tokens"] > metadata["config"]["runtime"]["max_sequence_tokens"]:
            raise ValueError(f"Token budget exceeded for {case.id}")
    expected = summarize(traces)
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(f"Summary accounting differs for {key}")
    if artifact_bytes(output) != summary["artifact_bytes"]:
        raise ValueError("Artifact byte total differs")
    if len(list((output / "traces").glob("*.json"))) != len(data["cases"]):
        raise ValueError("Trace count differs from manifest")
    return {"audit": "passed", **expected, "artifact_bytes": artifact_bytes(output)}
