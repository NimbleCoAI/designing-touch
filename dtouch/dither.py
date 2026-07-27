"""Dithering algorithms — Floyd-Steinberg error diffusion and Bayer ordered dithering.

Both algorithms are public-domain; these are independent reimplementations.
Inputs/outputs are float32 arrays in [0, 1]. Works on grayscale (H, W) or
colour (H, W, C) arrays.
"""
from __future__ import annotations

import numpy as np


def _bayer_matrix(n: int) -> np.ndarray:
    """Recursively generate a normalised n×n Bayer threshold matrix.

    n must be a power of 2. If sub = M(n)/n² is the normalised n×n matrix,
    the normalised 2n×2n matrix is:

        [[sub,        sub + 2/N²],
         [sub + 3/N², sub + 1/N²]]    where N² = (2n)²

    This is derived from the standard unnormalised recursion
    M(2n) = [[4·M(n), 4·M(n)+2], [4·M(n)+3, 4·M(n)+1]] by dividing both
    sides by N² = 4n² and using sub = M(n)/n².
    """
    if n == 1:
        return np.array([[0.0]], dtype=np.float32)
    sub = _bayer_matrix(n // 2)    # already normalised by (n//2)²
    N2 = float(n * n)              # normalisation factor for the output matrix
    return np.block([
        [sub,             sub + 2.0 / N2],
        [sub + 3.0 / N2,  sub + 1.0 / N2],
    ]).astype(np.float32)


def bayer_dither(img: np.ndarray, bits: int = 2, matrix_size: int = 4) -> np.ndarray:
    """Ordered dithering via Bayer threshold matrix.

    Fast and fully vectorised — suitable for real-time use. Preserves exact
    black (0.0) and white (1.0): the threshold bias is additive in the
    quantisation domain so floor(1.0 * levels + t) / levels = 1 for all t<1.

    Parameters
    ----------
    img : float32 (H, W) or (H, W, C) in [0, 1]
    bits : output bit depth per channel (1–8)
    matrix_size : Bayer matrix side length; must be a power of 2 (2, 4, 8, …)

    Returns
    -------
    float32, same shape as *img*, values quantised to ``2**bits`` levels.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be 1–8, got {bits}")
    if matrix_size < 1 or (matrix_size & (matrix_size - 1)) != 0:
        raise ValueError(f"matrix_size must be a power of 2, got {matrix_size}")

    levels = float((1 << bits) - 1)
    mat = _bayer_matrix(matrix_size)       # (S, S) values in [0, 1)
    h, w = img.shape[:2]
    ty = int(np.ceil(h / matrix_size))
    tx = int(np.ceil(w / matrix_size))
    threshold = np.tile(mat, (ty, tx))[:h, :w]   # (H, W) in [0, 1)

    if img.ndim == 3:
        threshold = threshold[:, :, np.newaxis]
    # floor(p * levels + threshold) maps uniformly-distributed threshold in [0,1)
    # to a dithering pattern that preserves p=0 → 0 and p=1 → 1 exactly.
    result = np.floor(img.astype(np.float32) * levels + threshold) / levels
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def floyd_steinberg(img: np.ndarray, bits: int = 2) -> np.ndarray:
    """Floyd-Steinberg error-diffusion dithering.

    Distributes quantisation error to four right/below neighbours using the
    classic 7/16 · 3/16 · 5/16 · 1/16 kernel. Sequential by nature — best
    suited to small images or offline use; use :func:`bayer_dither` for
    real-time paths.

    Parameters
    ----------
    img : float32 (H, W) or (H, W, C) in [0, 1]
    bits : output bit depth per channel (1–8)

    Returns
    -------
    float32, same shape as *img*, dithered.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be 1–8, got {bits}")

    levels = float((1 << bits) - 1)
    squeezed = img.ndim == 2
    buf = img.astype(np.float32)
    if squeezed:
        buf = buf[:, :, np.newaxis]
    buf = buf.copy()
    h, w, _ = buf.shape

    for y in range(h):
        for x in range(w):
            # .copy() is essential: buf[y, x] is a view; writing new back to
            # buf[y, x] would clobber old through the same view, zeroing err.
            old = buf[y, x].copy()
            new = np.round(old * levels) / levels
            buf[y, x] = new
            err = old - new
            if x + 1 < w:
                buf[y, x + 1]     += err * (7.0 / 16.0)
            if y + 1 < h:
                if x > 0:
                    buf[y + 1, x - 1] += err * (3.0 / 16.0)
                buf[y + 1, x]     += err * (5.0 / 16.0)
                if x + 1 < w:
                    buf[y + 1, x + 1] += err * (1.0 / 16.0)

    buf = np.clip(buf, 0.0, 1.0)
    return (buf[:, :, 0] if squeezed else buf).astype(np.float32)
