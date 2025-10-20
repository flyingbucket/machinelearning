from LBP import LBP, LBPcython
import time
import os
import numpy as np
from collections import Counter


def _to_hist256(x):
    """把 ndarray / dict / Counter 统一转成长度256的np.int64直方图。"""
    if isinstance(x, np.ndarray):
        if x.shape == (256,):
            return x.astype(np.int64, copy=False)
        h = np.zeros(256, dtype=np.int64)
        n = min(256, x.shape[0])
        h[:n] = x[:n]
        return h
    # dict / Counter
    h = np.zeros(256, dtype=np.int64)
    for k, v in dict(x).items():
        if 0 <= int(k) < 256:
            h[int(k)] = int(v)
    return h


def benchmark_lbp(im_path, pad=1, mode="reflect", runs=5, set_threads=None):
    # 可选：控制 OpenMP 线程数（不传则沿用当前环境）
    if set_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(set_threads)

    # 预读一次，拿到窗口数
    arr = LBP._read_img(im_path, pad=pad, mode=mode)
    H, W = arr.shape
    n_windows = (H - 2) * (W - 2)
    megapixels = n_windows / 1e6

    print(
        f"[Info] image shape={H}x{W}, pad={pad}, windows={n_windows} (~{megapixels:.3f} Mpx)"
    )
    if (H < 3) or (W < 3):
        raise ValueError("Image too small (<3x3) after padding")

    # 实例
    py_exec = LBP()
    cy_exec = LBPcython()  # 你已有的 Cython 封装类

    # ---- warm-up ----
    _ = py_exec(im_path, pad=pad, mode=mode)
    _ = cy_exec(im_path, pad=pad, mode=mode)

    # ---- Python LBP ----
    t_list = []
    hist_py = None
    for _ in range(runs):
        t0 = time.perf_counter()
        res = py_exec(im_path, pad=pad, mode=mode)
        t1 = time.perf_counter()
        t_list.append(t1 - t0)
        if hist_py is None:
            hist_py = _to_hist256(res)
    t_py = sum(t_list) / len(t_list)

    # ---- Cython LBP ----
    t_list = []
    hist_cy = None
    for _ in range(runs):
        t0 = time.perf_counter()
        res = cy_exec(im_path, pad=pad, mode=mode)
        t1 = time.perf_counter()
        t_list.append(t1 - t0)
        if hist_cy is None:
            hist_cy = _to_hist256(res)
    t_cy = sum(t_list) / len(t_list)

    # 校验
    same_hist = np.array_equal(hist_py, hist_cy)
    same_sum = int(hist_py.sum()) == int(hist_cy.sum()) == n_windows

    # 吞吐量（窗口/秒、Mpx/s）
    tp_py = n_windows / t_py
    tp_cy = n_windows / t_cy
    mp_py = tp_py / 1e6
    mp_cy = tp_cy / 1e6
    speedup = t_py / t_cy if t_cy > 0 else float("inf")

    print("\n=== LBP Performance ===")
    print(
        f"Python  : {t_py:.6f} s  | throughput: {tp_py:,.0f} win/s ({mp_py:.2f} Mpx/s)"
    )
    print(
        f"Cython  : {t_cy:.6f} s  | throughput: {tp_cy:,.0f} win/s ({mp_cy:.2f} Mpx/s)"
    )
    print(f"Speedup : ×{speedup:.2f}")
    print(
        f"Hist equal: {same_hist}  | sum check: {same_sum} (sum_py={hist_py.sum()}, sum_cy={hist_cy.sum()})"
    )

    # 若直方图不完全一致，打印前几个差异 bin 以便排查
    if not same_hist:
        diff = np.where(hist_py != hist_cy)[0]
        if diff.size:
            show = diff[:10]
            print("[Warn] first differing bins:", list(map(int, show)))
            for k in show:
                print(f"  bin {int(k):3d}: py={int(hist_py[k])}  cy={int(hist_cy[k])}")


if __name__ == "__main__":
    im_path = "./LBPtest_image.png"
    LBPExecutor = LBP()
    LBPcyExecutor = LBPcython()

    # 跑基准：可改 runs / 线程数
    benchmark_lbp(im_path, pad=1, mode="reflect", runs=5, set_threads=None)
