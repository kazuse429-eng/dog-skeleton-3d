# -*- coding: utf-8 -*-
"""距離場を格子にサンプリングする。

全身を一気に評価するとメモリが足りないので、2つの工夫をする。

1. **X 方向のスラブに切る。** 帯ごとに評価して使い終わったら捨てる。
2. **プリミティブごとに自分の境界箱の中だけ評価する。** 骨1本は全身の
   ごく一部しか占めないので、これで計算量が2桁落ちる。

ナローバンド
------------
等値面のまわり `band` mm の範囲だけ正しい値が要る。それより遠い所は
「十分遠い」ことさえ分かればよいので `FAR` で埋める。
OpenVDB に渡すときも、この FAR を背景値にすれば疎なグリッドになる。
"""

import numpy as np

from .prim import FAR, FLOAT
from .ops import Difference, Intersection, Mirror, Offset, SmoothUnion, Union


class Grid(object):
    """等方の直方格子。世界座標 mm。

    格子点 (i,j,k) の世界座標は origin + voxel * (i,j,k)。
    """

    def __init__(self, lo, hi, voxel, pad=None):
        self.voxel = float(voxel)
        if self.voxel <= 0:
            raise ValueError("voxel が 0 以下")
        pad = self.voxel * 4.0 if pad is None else float(pad)
        lo = np.asarray(lo, dtype=np.float64) - pad
        hi = np.asarray(hi, dtype=np.float64) + pad
        self.origin = lo
        n = np.ceil((hi - lo) / self.voxel).astype(int) + 1
        self.shape = tuple(int(v) for v in np.maximum(n, 2))

    def __repr__(self):
        n = self.shape
        return "Grid(%dx%dx%d voxel=%.3fmm %.1fMvox origin=%s)" % (
            n[0], n[1], n[2], self.voxel, n[0] * n[1] * n[2] / 1e6,
            np.round(self.origin, 1))

    @property
    def count(self):
        return self.shape[0] * self.shape[1] * self.shape[2]

    def coords(self, axis, i0=0, i1=None):
        i1 = self.shape[axis] if i1 is None else i1
        return (self.origin[axis]
                + self.voxel * np.arange(i0, i1, dtype=np.float64)).astype(FLOAT)

    def index_range(self, lo, hi):
        """世界座標の箱 [lo,hi] を含む格子インデックス範囲 (i0, i1) を軸ごとに。

        i1 は排他。格子の外にはみ出す分は切り詰める。
        重なりが無い軸があれば None を返す。
        """
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        i0 = np.floor((lo - self.origin) / self.voxel).astype(int)
        i1 = np.ceil((hi - self.origin) / self.voxel).astype(int) + 1
        n = np.array(self.shape)
        i0 = np.clip(i0, 0, n)
        i1 = np.clip(i1, 0, n)
        if np.any(i1 <= i0):
            return None
        return (i0, i1)

    def world_bbox(self):
        hi = self.origin + self.voxel * (np.array(self.shape) - 1)
        return (self.origin.copy(), hi)


def _points(gx, gy, gz):
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def _is_plain(node):
    """変換が入っていない合成ノードか（間引きの再帰を続けてよいか）。"""
    return (np.allclose(node.R, np.eye(3), atol=1e-12)
            and np.allclose(node.t, 0.0, atol=1e-12))


class _SlabView(object):
    """スラブ（X 方向の帯）に対する部分格子。"""

    def __init__(self, grid, i0, i1):
        self.grid = grid
        self.i0, self.i1 = int(i0), int(i1)
        self.shape = (self.i1 - self.i0, grid.shape[1], grid.shape[2])

    def sub_range(self, lo, hi):
        r = self.grid.index_range(lo, hi)
        if r is None:
            return None
        a, b = r
        a = a.copy()
        b = b.copy()
        a[0] = max(a[0], self.i0)
        b[0] = min(b[0], self.i1)
        if np.any(b <= a):
            return None
        return (a, b)

    def coords_for(self, a, b):
        g = self.grid
        return (g.coords(0, a[0], b[0]),
                g.coords(1, a[1], b[1]),
                g.coords(2, a[2], b[2]))

    def local_slice(self, a, b):
        return (slice(a[0] - self.i0, b[0] - self.i0),
                slice(a[1], b[1]), slice(a[2], b[2]))


def _eval_dense(node, view, a, b):
    gx, gy, gz = view.coords_for(a, b)
    P = _points(gx, gy, gz)
    d = node.dist(P).astype(FLOAT, copy=False)
    return d.reshape(len(gx), len(gy), len(gz))


def _accumulate(node, out, view, band):
    """node の距離を out に min で流し込む（out は FAR で初期化済み）。"""
    if isinstance(node, Union) and _is_plain(node):
        for c in node.children:
            _accumulate(c, out, view, band)
        return

    if isinstance(node, SmoothUnion) and _is_plain(node):
        # smin は k の範囲でしか効かないので、その分だけ広げて間引く
        _accumulate_smooth(node, out, view, band)
        return

    if isinstance(node, Difference) and _is_plain(node) and len(node.children) > 1:
        # **削る側も間引く。** 差は max(a, -b) なので、b の境界箱から
        # band 以上離れた所では -b が十分小さく、結果は a のまま変わらない。
        # 間引かないと、溝や穴のひとつひとつを全格子で評価することになる。
        _accumulate(node.children[0], out, view, band)
        for c in node.children[1:]:
            lo, hi = c.aabb()
            r = view.sub_range(lo - band, hi + band)
            if r is None:
                continue
            a, b = r
            d = _eval_dense(c, view, a, b)
            sl = view.local_slice(a, b)
            np.maximum(out[sl], -d, out=out[sl])
        return

    lo, hi = node.aabb()
    r = view.sub_range(lo - band, hi + band)
    if r is None:
        return
    a, b = r
    d = _eval_dense(node, view, a, b)
    sl = view.local_slice(a, b)
    np.minimum(out[sl], d, out=out[sl])


def _accumulate_smooth(node, out, view, band):
    k = FLOAT(node.k)
    for c in node.children:
        lo, hi = c.aabb()
        r = view.sub_range(lo - (band + node.k), hi + (band + node.k))
        if r is None:
            continue
        a, b = r
        d = _eval_dense(c, view, a, b)
        sl = view.local_slice(a, b)
        cur = out[sl]
        h = np.clip(FLOAT(0.5) + FLOAT(0.5) * (d - cur) / k,
                    FLOAT(0.0), FLOAT(1.0))
        out[sl] = (d * (FLOAT(1.0) - h) + cur * h) - k * h * (FLOAT(1.0) - h)


def sample_slab(field, grid, i0, i1, band):
    """grid の X インデックス [i0,i1) を評価して (i1-i0, ny, nz) を返す。"""
    view = _SlabView(grid, i0, i1)
    out = np.full(view.shape, FAR, dtype=FLOAT)
    _accumulate(field, out, view, float(band))
    return out


def slabs(grid, width=64):
    """スラブの (i0, i1) を順に返す。"""
    n = grid.shape[0]
    for i0 in range(0, n, width):
        yield (i0, min(i0 + width, n))


def sample(field, grid, band, slab_width=64, progress=None):
    """全体を1つの配列に評価する。**小さいモデルの検算用。**

    全身には使わないこと（メモリが要る）。全身は `mesh.py` が
    スラブごとに OpenVDB へ流し込む。
    """
    out = np.empty(grid.shape, dtype=FLOAT)
    for i0, i1 in slabs(grid, slab_width):
        out[i0:i1] = sample_slab(field, grid, i0, i1, band)
        if progress:
            progress(i1, grid.shape[0])
    return out


def grid_for(field, voxel, band=None, pad=None):
    """場の境界箱から格子を作る。"""
    lo, hi = field.aabb()
    band = voxel * 3.0 if band is None else band
    pad = band + voxel * 2.0 if pad is None else pad
    return Grid(lo, hi, voxel, pad=pad)
