"""The local Qwen3 models, pinned by Hugging Face commit, and how to serve each one."""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalModel:
    key: str
    repo: str
    revision: str
    port: int
    wired_gb: float

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def path(self) -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(self.repo, revision=self.revision, local_files_only=True)


MODELS = {
    m.key: m
    for m in [
        LocalModel("qwen3-4b", "mlx-community/Qwen3-4B-4bit", "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25", 8104, 4),
        LocalModel("qwen3-8b", "mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192", 8108, 6),
        LocalModel("qwen3-14b", "mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4", 8114, 10),
        LocalModel(
            "qwen3-30b-a3b-instruct",
            "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
            "e9675aa3ca5f900ccef55267914466d55ab325fa",
            8130,
            18,
        ),
    ]
}

PRIMITIVE_RUNNER = "qwen3-8b"
USER_SIMULATOR = "qwen3-30b-a3b-instruct"
STUDENTS = ("qwen3-4b", "qwen3-8b", "qwen3-14b")


def download() -> None:
    from huggingface_hub import snapshot_download

    for m in MODELS.values():
        print(m.key, snapshot_download(m.repo, revision=m.revision), flush=True)


def serve(key: str, concurrency: int, cache_gb: float) -> None:
    m = MODELS[key]
    cmd = [
        sys.executable, "-m", "plvr.serve",
        "--model", m.path(),
        "--port", str(m.port),
        "--max-tokens", "8192",
        "--decode-concurrency", str(concurrency),
        "--prompt-concurrency", str(concurrency),
        "--prompt-cache-bytes", str(int(cache_gb * 2**30)),
        "--log-level", "WARNING",
        "--wired-gb", str(m.wired_gb),
    ]
    os.execvp(cmd[0], cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    s = sub.add_parser("serve")
    s.add_argument("key", choices=sorted(MODELS))
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--cache-gb", type=float, default=2.0)
    args = parser.parse_args()
    if args.cmd == "download":
        download()
    else:
        serve(args.key, args.concurrency, args.cache_gb)


if __name__ == "__main__":
    main()
