import numpy as np
from PIL import Image


@staticmethod
def _read_img(imPath: str, pad: int = 1, mode: str = "reflect") -> np.ndarray:
    im = Image.open(imPath).convert("L")
    arr = np.array(im)
    padded = np.pad(arr, pad_width=(pad, pad), mode=mode)
    return padded


@staticmethod
def LBPkernel(im: np.ndarray, x, y) -> int:
    h, w = im.shape
    assert x + 2 < h and y + 2 < w, (
        f"Index out of bound,please check padding. x:{x},y:{y},h:{h},w:{w}"
    )
    patch = im[x : x + 3, y : y + 3].copy()
    patch = (patch >= patch[1, 1]).astype(np.uint8)
    idxs = [0, 1, 3, 5, 8, 7, 6, 3]
    bits = patch.reshape(-1)[idxs]
    val = int("".join(map(str, bits)), 2)
    return val


if __name__ == "__main__":
    im = np.array([[10, 20, 30], [5, 15, 25], [0, 10, 20]])

    print(LBPkernel(im, 0, 0))
