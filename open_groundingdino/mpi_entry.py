# open_groundingdino/mpi_entry.py
import os
import importlib
import pathlib
import runpy

def _dist_env():
    # OpenMPI 常用环境变量；没有则退化为单进程
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("LOCAL_RANK", str(rank))))
    return rank, world_size, local_rank

def _set_cvd_if_needed(local_rank: int):
    # 只在用户没显式指定 CUDA_VISIBLE_DEVICES 时才设置
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

def _torch_import_under_global_lock(rank: int, world_size: int):
    """
    关键：用文件锁把“首次 import torch（触发 jtorch/jittor 编译写 cache）”串行化。
    这样不会出现多个 rank 同时写 ~/.cache/jittor/... 导致源码文件损坏。
    """
    if world_size <= 1:
        return

    import fcntl
    lock_path = os.path.join(os.path.expanduser("~"), ".cache", "jittor", ".torch_jtorch_import.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    with open(lock_path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        if rank == 0:
            print(f"[rank {rank}] torch/jtorch warmup import under lock ...", flush=True)
        importlib.import_module("torch")
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def main():
    rank, world_size, local_rank = _dist_env()
    _set_cvd_if_needed(local_rank)
    _torch_import_under_global_lock(rank, world_size)

    # 进入你原来的 main.py（保持参数/argv 不变）
    target = pathlib.Path(__file__).with_name("main.py")
    runpy.run_path(str(target), run_name="__main__")

if __name__ == "__main__":
    main()