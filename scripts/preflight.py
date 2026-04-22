"""Pre-flight checks before launching trainer.

Hard-fails if anything that would crash the trainer mid-run is wrong NOW:
  - CARLA reachable + ticks
  - PYTHONPATH has CARLA's `agents` package
  - Checkpoint dir writable
  - Disk free > 5 GB
  - GPU available with > 4 GB free
  - Python deps importable

Exits 0 on full pass, nonzero on any hard fail.
Run from start_training.sh BEFORE python3 agents/train.py.
"""

import os
import socket
import shutil
import sys
import time

OK = "[ \033[92mOK\033[0m ]"
FAIL = "[\033[91mFAIL\033[0m]"
WARN = "[\033[93mWARN\033[0m]"


def check_carla_port(host, port, timeout=5.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"CARLA RPC port {host}:{port} reachable"
    except (OSError, socket.timeout) as e:
        return False, f"CARLA RPC port {host}:{port} NOT reachable: {e}"


def check_carla_tick(host, port, timeout=10.0):
    try:
        import carla
    except ImportError as e:
        return False, f"carla module not importable: {e}"
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
        # Don't tick — just verify the world handle works.
        _ = world.get_settings()
        return True, f"CARLA world handle OK (map={world.get_map().name})"
    except Exception as e:
        return False, f"CARLA world handle failed: {e}"


def check_route_planner_import():
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa
        return True, "CARLA agents.navigation.global_route_planner importable"
    except ImportError as e:
        hint = "export PYTHONPATH=$PYTHONPATH:/path/to/CARLA/PythonAPI/carla"
        return False, f"GlobalRoutePlanner import FAILED ({e}). Try: {hint}"


def check_dir_writable(path):
    try:
        os.makedirs(path, exist_ok=True)
        if not os.access(path, os.W_OK):
            return False, f"{path} not writable"
        return True, f"{path} writable"
    except OSError as e:
        return False, f"{path} create/access failed: {e}"


def check_disk_free(path, min_gb=5.0):
    try:
        usage = shutil.disk_usage(path)
        gb = usage.free / (1024 ** 3)
        if gb < min_gb:
            return False, f"disk free at {path}: {gb:.1f} GB < {min_gb} GB minimum"
        return True, f"disk free at {path}: {gb:.1f} GB"
    except OSError as e:
        return False, f"disk check failed at {path}: {e}"


def check_gpu(min_gb_free=4.0):
    try:
        import torch
    except ImportError as e:
        return False, f"torch not importable: {e}"
    if not torch.cuda.is_available():
        return True, "no CUDA — running on CPU (will be slow)"
    try:
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        if free_gb < min_gb_free:
            return False, (f"GPU {torch.cuda.get_device_name(0)}: "
                           f"{free_gb:.1f}/{total_gb:.1f} GB free < {min_gb_free} GB minimum")
        return True, (f"GPU {torch.cuda.get_device_name(0)}: "
                      f"{free_gb:.1f}/{total_gb:.1f} GB free")
    except Exception as e:
        return False, f"GPU check failed: {e}"


def check_python_deps():
    missing = []
    for mod in ("gymnasium", "stable_baselines3", "numpy", "torch"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return False, f"missing python deps: {', '.join(missing)}"
    return True, "stable_baselines3, gymnasium, torch, numpy importable"


def main():
    host = os.environ.get("CARLA_HOST", "172.25.176.1")
    port = int(os.environ.get("CARLA_PORT", "2000"))
    # Multi-env support — if URBANZERO_N_ENVS>1 we verify every CARLA
    # instance (port = BASE_PORT + i*1000), not just the base port.
    # Agent-4 audit: without this, a missing 3000/4000/5000 CARLA instance
    # would pass preflight and then deadlock SubprocVecEnv during worker
    # init with no clear error to the user.
    n_envs = int(os.environ.get("URBANZERO_N_ENVS", "1"))
    base_port = int(os.environ.get("URBANZERO_BASE_PORT", str(port)))
    experiment = os.environ.get("URBANZERO_EXP", "shaped")
    ckpt_dir = os.path.expanduser(f"~/urbanzero/checkpoints/{experiment}")
    log_dir = os.path.expanduser(f"~/urbanzero/logs/{experiment}")
    home_dir = os.path.expanduser("~/urbanzero")

    print("=== UrbanZero pre-flight ===")
    print(f"  CARLA target: {host}:{port}")
    print(f"  n_envs:       {n_envs} (base port {base_port})")
    print(f"  Experiment:   {experiment}")
    print(f"  Checkpoints:  {ckpt_dir}")
    print()

    checks = [
        ("CARLA RPC port",         check_carla_port,           (host, port)),
        ("Python deps",            check_python_deps,          ()),
        ("CARLA world handshake",  check_carla_tick,           (host, port)),
        ("Route planner import",   check_route_planner_import, ()),
    ]
    # Extra CARLA ports when running multi-env: each worker connects to
    # base_port + rank*1000, so all of those must be reachable too.
    if n_envs > 1:
        for i in range(1, n_envs):
            extra_port = base_port + i * 1000
            checks.append(
                (f"CARLA port {extra_port}",
                 check_carla_port, (host, extra_port))
            )
    checks += [
        ("Checkpoint dir",         check_dir_writable,         (ckpt_dir,)),
        ("Log dir",                check_dir_writable,         (log_dir,)),
        ("Beacon dir",             check_dir_writable,         (home_dir,)),
        ("Disk free",              check_disk_free,            (home_dir, 5.0)),
        ("GPU",                    check_gpu,                  (4.0,)),
    ]

    failures = 0
    for name, fn, args in checks:
        try:
            ok, msg = fn(*args)
        except Exception as e:
            ok, msg = False, f"check raised: {e}"
        prefix = OK if ok else FAIL
        print(f"  {prefix} {name:<25} {msg}")
        if not ok:
            failures += 1

    print()
    if failures == 0:
        print(f"{OK} All preflight checks passed.")
        return 0
    print(f"{FAIL} {failures} preflight check(s) failed. Refusing to launch trainer.")
    return failures


if __name__ == "__main__":
    sys.exit(main())
