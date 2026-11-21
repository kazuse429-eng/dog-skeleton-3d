# -*- coding: utf-8 -*-
"""距離場どうしの演算。

合成した結果もまた距離場なので、そのまま次の演算に渡せる。
**一体型が構造的に保証されるのはこの性質のおかげ**で、
ブーリアンのように「失敗する」ことがない。

`Union` と `SmoothUnion` は `children` を公開しており、
`grid.py` はそれを見て「子の境界箱の中だけ評価する」間引きをする。
全身を一気に評価しないで済むのはこのため。
"""

import numpy as np

from .prim import FLOAT, Field


class _Composite(Field):
    def __init__(self, children, **kw):
        Field.__init__(self, **kw)
        self.children = list(children)
        if not self.children:
            raise ValueError("子が無い")


class Union(_Composite):
    """和。距離は厳密に保たれる。"""

    def _local_dist(self, P):
        out = self.children[0].dist(P)
        for c in self.children[1:]:
            np.minimum(out, c.dist(P), out=out)
        return out

    def _local_aabb(self):
        los, his = zip(*[c.aabb() for c in self.children])
        return (np.min(np.array(los), axis=0), np.max(np.array(his), axis=0))


class SmoothUnion(_Composite):
    """滑らかな和。ブレンド半径 k で継ぎ目に丸みがつく。

    **展示骨格標本風では k を小さく取る。** 大きくすると関節が埋まって
    「溶けた」見た目になり、標本らしさが消える。
    骨どうしを確実に繋ぐのが目的なので、必要最小限でよい。

    注意: 多項式 smin は結合則を満たさないので、
    3つ以上を一度に混ぜると順序でわずかに結果が変わる。
    関節ごとに2つずつ混ぜる使い方を守ること。

    また smin は両者より小さい値を返しうるので、**距離が最大 k/4 だけ
    短く出る**（Lipschitz は保たれる）。肉厚の測定は安全側にずれる。
    """

    def __init__(self, children, k=3.0, **kw):
        _Composite.__init__(self, children, **kw)
        self.k = float(k)
        if self.k <= 0:
            raise ValueError("ブレンド半径 k が 0 以下。Union を使うこと")

    def _local_dist(self, P):
        k = FLOAT(self.k)
        out = self.children[0].dist(P)
        for c in self.children[1:]:
            b = c.dist(P)
            h = np.clip(FLOAT(0.5) + FLOAT(0.5) * (b - out) / k,
                        FLOAT(0.0), FLOAT(1.0))
            out = (b * (FLOAT(1.0) - h) + out * h) - k * h * (FLOAT(1.0) - h)
        return out

    def _local_aabb(self):
        los, his = zip(*[c.aabb() for c in self.children])
        # ブレンドで k/4 だけ外へ膨らむ
        pad = self.k * 0.25
        return (np.min(np.array(los), axis=0) - pad,
                np.max(np.array(his), axis=0) + pad)


class Difference(_Composite):
    """差。children[0] から残りを引く。眼窩・閉鎖孔・椎孔に使う。"""

    def _local_dist(self, P):
        out = self.children[0].dist(P)
        for c in self.children[1:]:
            np.maximum(out, -c.dist(P), out=out)
        return out

    def _local_aabb(self):
        return self.children[0].aabb()


class Intersection(_Composite):
    """積。板を面で切り落とすときなどに使う。"""

    def _local_dist(self, P):
        out = self.children[0].dist(P)
        for c in self.children[1:]:
            np.maximum(out, c.dist(P), out=out)
        return out

    def _local_aabb(self):
        los, his = zip(*[c.aabb() for c in self.children])
        lo = np.max(np.array(los), axis=0)
        hi = np.min(np.array(his), axis=0)
        if np.any(lo > hi):                      # 重なりが無い
            mid = (lo + hi) * 0.5
            return (mid, mid)
        return (lo, hi)


class Offset(Field):
    """一様に太らせる（r>0）／痩せさせる（r<0）。

    **印刷下限を満たすための太らせはこれで行う。**
    距離場を平行移動するだけなので、既存版の平滑化のように
    「幅を勝手に削る」方向には決して働かない。
    """

    def __init__(self, child, r, **kw):
        Field.__init__(self, **kw)
        self.child = child
        self.r = float(r)

    def _local_dist(self, P):
        return self.child.dist(P) - FLOAT(self.r)

    def _local_aabb(self):
        lo, hi = self.child.aabb()
        return (lo - max(0.0, self.r), hi + max(0.0, self.r))


class Mirror(Field):
    """XZ 平面（y=0）に対する鏡像を足す。左右対の骨に使う。

    `both=True` なら元と鏡像の和、False なら鏡像だけ。
    """

    def __init__(self, child, both=True, **kw):
        Field.__init__(self, **kw)
        self.child = child
        self.both = bool(both)

    def _local_dist(self, P):
        Q = P.copy()
        Q[:, 1] = -Q[:, 1]
        d = self.child.dist(Q)
        if self.both:
            np.minimum(d, self.child.dist(P), out=d)
        return d

    def _local_aabb(self):
        lo, hi = self.child.aabb()
        mlo = np.array([lo[0], -hi[1], lo[2]])
        mhi = np.array([hi[0], -lo[1], hi[2]])
        if not self.both:
            return (mlo, mhi)
        return (np.minimum(lo, mlo), np.maximum(hi, mhi))


def union(*children, **kw):
    """使いやすい入口。1つだけならそのまま返す。"""
    flat = []
    for c in children:
        if isinstance(c, (list, tuple)):
            flat.extend(c)
        elif c is not None:
            flat.append(c)
    if not flat:
        raise ValueError("子が無い")
    if len(flat) == 1 and not kw:
        return flat[0]
    return Union(flat, **kw)
