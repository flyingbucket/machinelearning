# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
from collections import Counter
from PIL import Image

cdef extern from "omp.h" nogil:
    int omp_get_thread_num()
    int omp_get_max_threads()

cpdef compute_lbp_hist(cnp.ndarray[cnp.uint8_t, ndim=2] arr):
    cdef int H = arr.shape[0]
    cdef int W = arr.shape[1]
    cdef cnp.uint8_t[:, :] A = arr  # typed memoryview 视图，零拷贝

    # 预声明所有在循环体里要用的局部变量
    cdef int x, y, code
    cdef int tid, k, th, nthreads
    cdef cnp.uint8_t c, v
    cdef cnp.int64_t[:, :] H2   # hist2d 的 typed memoryview
    cdef cnp.int64_t[:] hist_local
    cdef cnp.uint8_t[:] row0, row1, row2  # 可选：行指针（更快）

    # 邻域偏移表
    cdef int dx[8]
    cdef int dy[8]
    dx[0], dy[0] = 0, 0
    dx[1], dy[1] = 0, 1
    dx[2], dy[2] = 0, 2
    dx[3], dy[3] = 1, 2
    dx[4], dy[4] = 2, 2
    dx[5], dy[5] = 2, 1
    dx[6], dy[6] = 2, 0
    dx[7], dy[7] = 1, 0

    nthreads = omp_get_max_threads()
    cdef cnp.ndarray[cnp.int64_t, ndim=2] hist2d = np.zeros((nthreads, 256), dtype=np.int64)
    H2 = hist2d  # 绑定为 memoryview，后续 nogil 安全

    # 主循环：注意 prange 块里的变量都是“先声明、后赋值”
    for x in prange(0, H - 2,2, nogil=True, schedule='static'):
        tid = omp_get_thread_num()
        for y in range(0, W - 2,2):
            c = A[x + 1, y + 1]
            code = 0
            v = A[x + dx[0], y + dy[0]]; code = (code << 1) | (1 if v >= c else 0)
            v = A[x + dx[1], y + dy[1]]; code = (code << 1) | (1 if v >= c else 0)
            v = A[x + dx[2], y + dy[2]]; code = (code << 1) | (1 if v >= c else 0)
            v = A[x + dx[3], y + dy[3]]; code = (code << 1) | (1 if v >= c else 0)
            v = A[x + dx[4], y + dy[4]]; code = (code << 1) | (1 if v >= c else 0)
            v = A[x + dx[5], y + dy[5]]; code = (code << 1) | (1 if v >= c else 0)
            v = A[x + dx[6], y + dy[6]]; code = (code << 1) | (1 if v >= c else 0)
            v = A[x + dx[7], y + dy[7]]; code = (code << 1) | (1 if v >= c else 0)
            # hist_local[code] += 1
            H2[tid,code]+=1

    # 归并
    cdef cnp.ndarray[cnp.int64_t, ndim=1] hist = np.zeros((256,), dtype=np.int64)
    for th in range(nthreads):
        for k in range(256):
            hist[k] += hist2d[th, k]

    return hist
def LBPfunc_cython(imPath:str,pad: int = 1, mode: str = "reflect")->Counter[int]:
    im = Image.open(imPath).convert("L")
    arr = np.array(im)
    padded = np.pad(arr, pad_width=((pad, pad), (pad, pad)), mode=mode)
    hist_arr = compute_lbp_hist(padded)
    res = Counter({i:hist_arr[i] for i in range(256) if hist_arr[i]})
    return res
