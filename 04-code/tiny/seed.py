"""tiny/seed.py (Chapter 26): seeds, fingerprints and the run card.

A seed is the starting state of one random-number generator.  This module keeps one generator per job
(starting tables, reading order, dropout masks, sampling), each derived from one master seed and the job's
name, so that changing the reading order cannot move the starting tables and the other way round.

  streams(master)            -> dict of NumPy Generators, one per job in JOBS, from SeedSequence(master).spawn
  seed_everything(s, threads=1, deterministic=True)
                             -> seeds Python's random, NumPy's global generator and PyTorch's; sets the thread count
                                and asks PyTorch for deterministic kernels (warn_only=False); returns a run card
  fingerprint(tables)        -> short SHA-256 of every table's bytes in name order: two runs match bit for bit
                                exactly when their fingerprints match
  run_card(**extra)          -> the facts that decide whether a run can be repeated: library versions, threads,
                                dtype defaults, the deterministic flag, CPU, plus whatever the caller adds
  ulp_distance(a, b)         -> how many representable 32-bit numbers lie between a and b (0 = the same number)
"""
import os, random, hashlib, platform, json
import numpy as np

JOBS = ("init", "order", "dropout", "sample")


def streams(master=1865, jobs=JOBS):
    """One independent NumPy Generator per job, all from one master seed.  Adding a job at the end does not
    change the generators of the jobs before it (SeedSequence.spawn is position-based)."""
    kids = np.random.SeedSequence(master).spawn(len(jobs))
    return {j: np.random.default_rng(k) for j, k in zip(jobs, kids)}


def seed_everything(seed=1865, threads=1, deterministic=True):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    card = dict(seed=seed, threads=threads, deterministic=deterministic)
    try:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # needed by cuBLAS when deterministic on a GPU
    except ImportError:
        pass
    card.update(run_card())
    return card


def fingerprint(tables, n=16):
    """SHA-256 over the raw bytes of each table, taken in sorted name order; first n hex digits."""
    h = hashlib.sha256()
    for k in sorted(tables):
        v = tables[k]
        a = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:n]


def run_card(**extra):
    card = dict(python=platform.python_version(), numpy=np.__version__, machine=platform.machine(),
                processor=platform.processor() or "unknown", omp_threads=os.environ.get("OMP_NUM_THREADS"))
    try:
        import torch
        card.update(torch=torch.__version__, torch_threads=torch.get_num_threads(),
                    deterministic=torch.are_deterministic_algorithms_enabled(),
                    default_dtype=str(torch.get_default_dtype()))
    except ImportError:
        pass
    card.update(extra)
    return card


def ulp_distance(a, b):
    """Number of float32 steps between a and b (both converted to float32)."""
    ia = np.array(a, dtype=np.float32).view(np.int32).astype(np.int64)
    ib = np.array(b, dtype=np.float32).view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)        # map the sign-magnitude order onto a number line
    ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(ia - ib)
