from LBP import LBP, LBPcython, LBPfunc, LBPskimage
from liblocal.lbp_cy import LBPfunc_cython
import time
import os
import numpy as np
import argparse


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
    # n_windows = (H - 2) * (W - 2)

    n_windows_height = (H - 3) // 2 + 1  # 高度方向上的窗口数
    n_windows_width = (W - 3) // 2 + 1  # 宽度方向上的窗口数

    n_windows = n_windows_height * n_windows_width  # 总窗口数
    megapixels = n_windows / 1e6

    print(
        f"[Info] image shape={H}x{W}, pad={pad}, windows={n_windows} (~{megapixels:.3f} Mpx)"
    )
    if (H < 3) or (W < 3):
        raise ValueError("Image too small (<3x3) after padding")

    # 实例化类
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
    print(f"Time py: {t_py}")

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
    print(f"Time cy: {t_cy}")

    # ---- LBPfunc ----
    t_list = []
    hist_func = None
    for _ in range(runs):
        t0 = time.perf_counter()
        res = LBPfunc(im_path, pad=pad, mode=mode)  # Direct function call
        t1 = time.perf_counter()
        t_list.append(t1 - t0)
        if hist_func is None:
            hist_func = _to_hist256(res)
    t_func = sum(t_list) / len(t_list)
    print(f"Time py func: {t_func}")

    # ---- LBPfunc_cython ----
    t_list = []
    hist_func_cy = None
    for _ in range(runs):
        t0 = time.perf_counter()
        res = LBPfunc_cython(im_path, pad=pad, mode=mode)  # Direct function call
        t1 = time.perf_counter()
        t_list.append(t1 - t0)
        if hist_func_cy is None:
            hist_func_cy = _to_hist256(res)
    t_func_cy = sum(t_list) / len(t_list)
    print(f"Time cy func: {t_func_cy}")

    # ---- LBPskimage ----
    t_list = []
    hist_skimage = None
    for _ in range(runs):
        t0 = time.perf_counter()
        res = LBPskimage(im_path)  # Direct function call
        t1 = time.perf_counter()
        t_list.append(t1 - t0)
        if hist_skimage is None:
            hist_skimage = _to_hist256(res)
    t_skimage = sum(t_list) / len(t_list)
    print(f"Time skimage: {t_skimage}")

    # 校验
    same_hist = (
        np.array_equal(hist_py, hist_cy)
        and np.array_equal(hist_py, hist_func)
        and np.array_equal(hist_py, hist_func_cy)
    )

    same_sum = (
        int(hist_py.sum())
        == int(hist_cy.sum())
        == int(hist_func.sum())
        == int(hist_func_cy.sum())
        == n_windows
    )

    # 吞吐量（窗口/秒、Mpx/s）
    tp_py = n_windows / t_py
    tp_cy = n_windows / t_cy
    tp_func = n_windows / t_func
    tp_func_cy = n_windows / t_func_cy
    tp_skimage = n_windows / t_skimage
    mp_py = tp_py / 1e6
    mp_cy = tp_cy / 1e6
    mp_func = tp_func / 1e6
    mp_func_cy = tp_func_cy / 1e6
    mp_skimage = tp_skimage / 1e6

    speedup_cy = t_py / t_cy if t_cy > 0 else float("inf")
    speedup_func = t_py / t_func if t_func > 0 else float("inf")
    speedup_func_cy = t_py / t_func_cy if t_func_cy > 0 else float("inf")
    speedup_skimage = t_py / t_skimage if t_skimage > 0 else float("inf")

    print("\n=== LBP Performance ===")
    print(
        f"Python  : {t_py:.6f} s  | throughput: {tp_py:,.0f} win/s ({mp_py:.2f} Mpx/s)"
    )
    print(
        f"Cython  : {t_cy:.6f} s  | throughput: {tp_cy:,.0f} win/s ({mp_cy:.2f} Mpx/s)"
    )
    print(
        f"LBPfunc : {t_func:.6f} s  | throughput: {tp_func:,.0f} win/s ({mp_func:.2f} Mpx/s)"
    )
    print(
        f"LBPfunc_cython : {t_func_cy:.6f} s  | throughput: {tp_func_cy:,.0f} win/s ({mp_func_cy:.2f} Mpx/s)"
    )
    print(
        f"LBPskimage : {t_skimage:.6f} s  | throughput: {tp_skimage:,.0f} win/s ({mp_skimage:.2f} Mpx/s)"
    )
    print(f"Speedup (cy): ×{speedup_cy:.2f}")
    print(f"Speedup (func): ×{speedup_func:.2f}")
    print(f"Speedup (cython func): ×{speedup_func_cy:.2f}")
    print(f"Speedup (skimage): ×{speedup_skimage:.2f}")
    print(
        f"Hist equal: {same_hist}  | sum check: {same_sum} (sum_py={hist_py.sum()}, sum_cy={hist_cy.sum()},sum_func={hist_func.sum()})"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--im_path", type=str, default="./LBPtest_image.png")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--num_threads", type=int, default=8)
    args = parser.parse_args()

    im_path = args.im_path
    runs = args.runs
    num_threads = args.num_threads

    LBPExecutor = LBP()
    LBPcyExecutor = LBPcython()
    LBPfuncExecutor = LBPfunc

    # 跑基准：可改 runs / 线程数
    benchmark_lbp(im_path, pad=1, mode="reflect", runs=runs, set_threads=num_threads)
