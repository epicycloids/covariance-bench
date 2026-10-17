"""Cryptographic digests, environment inspection, and deterministic seed derivation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

EXCLUDED = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "target"}


def digest(value):
    """Return the hex SHA-256 digest of a canonical JSON-serialized value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    """Return the hex SHA-256 digest of a file's raw bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(path):
    """Return a deterministic SHA-256 digest over all non-excluded files in ``path``."""
    root = Path(path)
    entries = [(str(p.relative_to(root)), file_hash(p)) for p in sorted(root.rglob("*"))
               if p.is_file() and not EXCLUDED.intersection(p.relative_to(root).parts)]
    return digest(entries)


def environment(threads):
    """Collect Python, OS, CPU, GPU, installed package, and thread-limit metadata."""
    cpu = platform.processor()
    info = Path("/proc/cpuinfo")
    if info.exists():
        cpu = next((line.partition(":")[2].strip() for line in info.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    gpu = None
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                       "--format=csv,noheader"], text=True, timeout=3).strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    affinity = sorted(os.sched_getaffinity(0))[:threads] if hasattr(os, "sched_getaffinity") else None
    return dict(python=platform.python_version(), platform=platform.platform(), cpu=cpu,
                logical_cpus=os.cpu_count(), requested_threads=threads, packages=packages,
                selected_cpu_affinity=affinity, visible_nvidia_devices=gpu,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                thread_environment={k: str(threads) for k in
                                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")})


def derived_seed(root_seed, scenario_id, replication):
    """Derive a deterministic 64-bit integer seed from ``(root_seed, scenario_id, replication)``."""
    return int(digest([root_seed, scenario_id, replication])[:16], 16)
