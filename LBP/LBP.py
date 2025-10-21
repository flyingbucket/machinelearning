import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from PIL import Image
from liblocal.LBP import lbp_cy


class LBP:
    @staticmethod
    def _read_img(imPath: str, pad: int = 1, mode: str = "reflect") -> np.ndarray:
        im = Image.open(imPath).convert("L")
        arr = np.array(im)
        padded = np.pad(arr, pad_width=((pad, pad), (pad, pad)), mode=mode)
        return padded

    @staticmethod
    def LBPkernel(im: np.ndarray, x, y) -> int:
        h, w = im.shape
        assert x + 2 < h and y + 2 < w, (
            f"Index out of bound,please check padding. x:{x},y:{y},h:{h},w:{w}"
        )
        patch = im[x : x + 3, y : y + 3].copy()
        patch = (patch >= patch[1, 1]).astype(np.uint8)
        idxs = [0, 1, 2, 5, 8, 7, 6, 3]
        bits = patch.reshape(-1)[idxs]
        val = int("".join(map(str, bits)), 2)
        return val

    def __call__(
        self, imPath: str, pad: int = 1, mode: str = "reflect"
    ) -> Counter[int]:
        arr = LBP._read_img(imPath, pad, mode)
        h, w = arr.shape
        res = Counter(
            LBP.LBPkernel(arr, x, y) for x in range(h - 2) for y in range(w - 2)
        )
        return res


def LBPfunc(imPath: str, pad: int = 1, mode: str = "reflect") -> Counter[int]:
    im = Image.open(imPath).convert("L")
    arr = np.array(im)
    pad_width = ((pad, pad), (pad, pad))
    arr = np.pad(arr, pad_width=pad_width, mode=mode)

    def LBPkernel(im: np.ndarray, x, y) -> int:
        h, w = im.shape
        assert x + 2 < h and y + 2 < w, (
            f"Index out of bound,please check padding. x:{x},y:{y},h:{h},w:{w}"
        )
        patch = im[x : x + 3, y : y + 3].copy()
        patch = (patch >= patch[1, 1]).astype(np.uint8)
        idxs = [0, 1, 2, 5, 8, 7, 6, 3]
        bits = patch.reshape(-1)[idxs]
        val = int("".join(map(str, bits)), 2)
        return val

    h, w = arr.shape
    res = Counter(LBPkernel(arr, x, y) for x in range(h - 2) for y in range(w - 2))
    return res


class LBPcython:
    @staticmethod
    def _read_img(imPath: str, pad: int = 1, mode: str = "reflect") -> np.ndarray:
        im = Image.open(imPath).convert("L")
        arr = np.array(im)
        padded = np.pad(arr, pad_width=((pad, pad), (pad, pad)), mode=mode)
        return padded

    def __call__(
        self, imPath: str, pad: int = 1, mode: str = "reflect"
    ) -> Counter[int]:
        arr = LBP._read_img(imPath, pad, mode).astype(np.uint8, copy=False)
        hist = lbp_cy.compute_lbp_hist(arr)  # ndarray shape=(256,), dtype=int64
        # 转 Counter（非零项）
        return Counter({i: int(hist[i]) for i in range(256) if hist[i]})


if __name__ == "__main__":
    im_path = "./LBPtest_image.png"
    LBPExecutor = LBP()
    LBPcyExecutor = LBPcython()
    res_dict = LBPcyExecutor(im_path)

    vals = list(res_dict.keys())
    counts = list(res_dict.values())

    fig, ax = plt.subplots()
    ax.bar(vals, counts, width=1.0)
    ax.set_xlabel("LBP code")
    ax.set_ylabel("Frequency")
    ax.set_title("LBP Histogram")
    plt.show()
