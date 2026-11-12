from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import marshal
from pathlib import Path

from .types import ToolRegistry


def canonical(value):
    if isinstance(value, ToolRegistry):
        return {name: canonical(spec) for name, spec in sorted(value.specs.items())}
    if callable(value):
        code = getattr(value, "__code__", None)
        return {
            "callable": f"{value.__module__}.{value.__qualname__}",
            "code": hashlib.sha256(marshal.dumps(code)).hexdigest() if code else None,
            "defaults": canonical(getattr(value, "__defaults__", None)),
            "keyword_defaults": canonical(getattr(value, "__kwdefaults__", None)),
            "closure": [canonical(cell.cell_contents) for cell in (getattr(value, "__closure__", None) or ())],
        }
    if is_dataclass(value):
        return {"record_type": type(value).__name__, **{f.name: canonical(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(canonical(v) for v in value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class VersionStamp:
    primitive_revision: str
    model_revision: str
    adapter_revision: str
    prompt_revision: str
    decoding: str
    runtime_policy: str
    executor_revision: str

    @classmethod
    def from_root(cls, root, model_config):
        root = Path(root)
        source = lambda name: hashlib.sha256((root / "src/plvr3b" / name).read_bytes()).hexdigest()
        return cls(
            digest([source("primitives/__init__.py"), source("contracts.py"), source("types.py")]),
            model_config["revision"], "none",
            digest({"oracle": "structured-oracle-v1", "template": model_config["chat_template"]}),
            digest(model_config["decoding"]), digest(model_config),
            digest([source("executor.py"), source("baseline_tasks.py")]),
        )


class RuntimeCache:
    def __init__(self, versions):
        self.versions = versions
        self.primitive = {}
        self.execution = {}
        self.events = []
        self.stats = {"primitive_hits": 0, "primitive_misses": 0, "execution_hits": 0, "execution_misses": 0, "invalidations": 0}

    def update_versions(self, versions):
        if versions != self.versions:
            self.versions = versions
            self.primitive.clear()
            self.execution.clear()
            self.stats["invalidations"] += 1
            self.events.append({"kind": "invalidate", "versions": canonical(versions)})

    def key(self, kind, payload):
        return digest({"kind": kind, "versions": self.versions, "payload": payload})

    def get(self, kind, key):
        store = getattr(self, kind)
        hit = key in store
        self.stats[f"{kind}_{'hits' if hit else 'misses'}"] += 1
        self.events.append({"kind": kind, "key": key, "hit": hit})
        return hit, deepcopy(store[key]) if hit else None

    def put(self, kind, key, value):
        getattr(self, kind)[key] = deepcopy(value)
