# -*- coding: utf-8 -*-
"""コーネル大学の犬の脳アトラス（MRI、30頭の平均）から、脳の部位ごとのメッシュを作る。

    python gen/brain_mesh.py <アトラスのフォルダ>

元データ: Johnson, P. J. & Barry, E. F. (2019) Cortical Atlas of the Canine Brain [dataset].
Cornell University Library eCommons. https://doi.org/10.7298/4t8z-aw34  （CC0）

アトラスの «脳葉» ラベルは皮質（灰白質）だけに付いていて、«皮質下» ラベルには
小脳・嗅球・深部の核がまとめて入っている。脳幹にはラベルが無い。そこで
  - 脳葉 1〜5 と帯状回はそのまま
  - 皮質下ラベルを位置で 小脳（後ろ）・嗅球（前）に分ける（深部の核は表示しない）
  - ラベルの無い脳の部分のうち、腹側・後方のつながった塊を脳幹とする
  - 残りの大脳の中身は «大脳髄質（白質）» として埋める（脳葉のすき間から中が抜けて見えないように）
出力は左半球だけ（ビューアで鏡像にして右を作る）。正中の構造（小脳・脳幹）は両側を含む。
座標はアトラスのままの mm（+X 頭側、+Y 左、+Z 背側、原点は脳の中心）。体への配置は realfit.py。
"""

import gzip
import os
import struct
import sys

import numpy as np
from scipy import ndimage
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "scan")


def load(fn):
    b = gzip.open(fn).read()
    dim = struct.unpack('<8h', b[40:56])
    dt = struct.unpack('<h', b[70:72])[0]
    pix = struct.unpack('<8f', b[76:108])
    off = int(struct.unpack('<f', b[108:112])[0])
    types = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 512: np.uint16}
    n = dim[1] * dim[2] * dim[3]
    a = np.frombuffer(b, dtype=types[dt], count=n, offset=off).reshape(dim[3], dim[2], dim[1]).transpose(2, 1, 0)
    return a, pix[1:4]


def taubin(V, T, n=12, lam=0.5, mu=-0.53):
    """形を縮めずに表面のボクセルの段差をならす。"""
    nb = [[] for _ in range(len(V))]
    for a, b, c in T:
        nb[a] += [b, c]; nb[b] += [a, c]; nb[c] += [a, b]
    idx = np.concatenate([np.unique(x) for x in nb])
    cnt = np.array([len(np.unique(x)) for x in nb])
    own = np.repeat(np.arange(len(V)), cnt)
    for _ in range(n):
        for f in (lam, mu):
            avg = np.zeros_like(V)
            np.add.at(avg, own, V[idx])
            avg /= np.maximum(cnt, 1)[:, None]
            V = V + f * (avg - V)
    return V


def mesh_of(mask, pix, sigma=1.0, step=2):
    f = ndimage.gaussian_filter(mask.astype(np.float32), sigma)
    V, T, _, _ = measure.marching_cubes(f, 0.5, spacing=pix, step_size=step)
    V = taubin(V, T)
    return V, T


def main():
    root = sys.argv[1]
    lobes, pix = load(os.path.join(root, "Lobe_masks", "all_lobes.nii.gz"))
    lobes = np.rint(lobes).astype(np.int16)
    lobes[lobes > 6] = 0
    cing, _ = load(os.path.join(root, "Lobe_masks", "cingulate_lobe_mask.nii.gz"))
    gm, _ = load(os.path.join(root, "TSMs", "gm_mask.nii.gz"))
    wm, _ = load(os.path.join(root, "TSMs", "wm_mask.nii.gz"))
    brain = (gm > 0) | (wm > 0)
    ni, nj, nk = brain.shape
    mid = 61                                  # 正中（i 方向）。ラベルの重心から
    K = np.arange(nk)[None, None, :]
    J = np.arange(nj)[None, :, None]

    sub = lobes == 6
    cereb = sub & (K < 62)
    olf = sub & (K > 150)
    # 脳幹: ラベル無しの脳のうち、腹側・後方
    unl = brain & (lobes == 0) & (cing == 0)
    stem = unl & (J < 40) & (K < 112) & (K > 8)
    lab, n = ndimage.label(stem)
    if n:
        sizes = ndimage.sum(stem, lab, range(1, n + 1))
        stem = lab == (1 + int(np.argmax(sizes)))
    cereb = cereb | (unl & (K < 50) & (J >= 40))          # 小脳の白質（小脳活樹）も小脳に
    cerebrum = brain & ~cereb & ~stem & ~olf
    white = cerebrum & (lobes == 0) & (cing == 0)

    left = np.zeros(brain.shape, dtype=bool)
    left[mid:] = True
    parts = {
        "brain_frontal": (lobes == 1) & left, "brain_occipital": (lobes == 2) & left,
        "brain_parietal": (lobes == 3) & left, "brain_temporal": (lobes == 4) & left,
        "brain_sensorimotor": (lobes == 5) & left, "brain_cingulate": (cing > 0) & left,
        "brain_olfactory": olf & left, "brain_white": white & left,
        "brain_cerebellum": cereb, "brain_stem": stem,
    }
    # 座標: i→左右(Y)、j→背腹(Z)、k→頭尾(X)。原点は脳の重心
    c = np.argwhere(brain).mean(0) * pix
    os.makedirs(OUT, exist_ok=True)
    for name, m in parts.items():
        V, T = mesh_of(m, pix)
        W = np.stack([V[:, 2] - c[2], V[:, 0] - c[0], V[:, 1] - c[1]], axis=1)
        # (i,j,k) → (k,i,j) は右手系のまま。面の向きを外向きに保つ
        np.savez_compressed(os.path.join(OUT, name + ".npz"), V=W.astype(np.float32), T=T.astype(np.int32))
        print("%-20s 体積 %7.0f mm3  三角 %6d" % (name, m.sum() * np.prod(pix), len(T)))


if __name__ == "__main__":
    main()
