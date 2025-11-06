import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from collections import Counter
from PIL import Image
from liblocal.lbp_cy import LBPfunc_cython, compute_lbp_hist
from skimage import feature


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
            LBP.LBPkernel(arr, x, y)
            for x in range(0, h - 2, 2)
            for y in range(0, w - 2, 2)
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
    res = Counter(
        LBPkernel(arr, x, y) for x in range(0, h - 2, 2) for y in range(0, w - 2, 2)
    )
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
        hist = compute_lbp_hist(arr)  # ndarray shape=(256,), dtype=int64
        return Counter({i: int(hist[i]) for i in range(256) if hist[i]})


def LBPskimage(imPath, pad=1, mode="reflect"):
    im = Image.open(imPath).convert("L")
    arr = np.array(im)
    padded = np.pad(arr, pad_width=((pad, pad), (pad, pad)), mode=mode)
    lbp = feature.local_binary_pattern(padded, P=8, R=1, method="default")
    lbp_flat = lbp.ravel().astype(int)
    lbp_hist = Counter(lbp_flat)
    return lbp_hist


def walk_dir(root_dir: str, out_dir: str = "EX1/outputs"):
    root = Path(root_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    LBPcyExecutor = LBPcython()
    # LBPcyExecutor = LBP()

    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        hist_list = []
        img_names = []
        all_codes = set()

        for img_path in sorted(class_dir.iterdir()):
            try:
                res_dict = LBPcyExecutor(str(img_path))  # {code: count}
                if not isinstance(res_dict, dict) or len(res_dict) == 0:
                    print(f"[WARN] 空直方图：{img_path}")
                    continue
                hist_list.append(res_dict)
                img_names.append(img_path.stem)
                all_codes.update(res_dict.keys())
            except Exception as e:
                print(f"[WARN] 处理失败: {img_path} -> {e}")

        if not hist_list:
            print(f"[INFO] 跳过空目录：{class_dir}")
            continue

        codes = sorted(all_codes)  # 所有出现过的 LBP code
        X = []  # 每张图对齐后的频率向量

        for h in hist_list:
            vec = np.array([h.get(c, 0) for c in codes], dtype=np.float64)
            X.append(vec)

        plt.figure(figsize=(10, 6))
        for vec, name in zip(X, img_names):
            plt.plot(codes, vec, linewidth=1.2, alpha=0.85, label=name)

        plt.xlabel("LBP code")
        plt.ylabel("count")
        plt.title(f"LBP feature curves - {class_dir.name}")
        plt.legend(ncol=2, fontsize=9, loc="best")
        plt.tight_layout()

        save_path = out / f"{class_dir.name}_lbp_curves.png"
        plt.savefig(save_path, dpi=160)
        plt.close()
        print(f"[OK] Saved: {save_path}")


if __name__ == "__main__":
    dir = "./EX1/data"
    walk_dir(dir)
