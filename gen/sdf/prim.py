# -*- coding: utf-8 -*-
"""距離場のプリミティブ。

このモジュールの約束
--------------------
- 単位は **mm（実寸）**。縮尺は最後に一度だけ掛ける。
- 各プリミティブは「その点から表面までの符号つき距離」を返す。
  内側が負、外側が正。
- 距離は**本物の距離**であること（Lipschitz 定数 1）。
  これが成り立っていれば、内部の値がそのまま
  「その点を中心に置ける最大の球の半径」になる。
  肉厚をレイキャストではなくこれで測るのが、この作り直しの一番の目的。

  厚みを場所によって変える板（`Plate` の `thickness` に関数を渡す場合）は
  この性質が厳密には崩れる。崩れ幅は `checks/test_sdf.py` が測る。

座標系
------
    +X = 頭側   +Y = 左   +Z = 上   Z=0 = 地面
（既存版 `lib/benv.py` と同じ。1 Blender unit = 1 mm）

すべてのプリミティブは「ローカル座標で定義し、`xform` で世界に置く」。
`xform` は (R, t) で、世界座標 = R @ ローカル + t。
"""

import math

import numpy as np

FLOAT = np.float32

#: 十分遠いことを表す距離。ナローバンドの外はこの値で埋める。
FAR = FLOAT(1.0e6)


# ── 変換 ──────────────────────────────────────────────────────────────

def identity():
    return (np.eye(3), np.zeros(3))


def rot_axis(axis, deg):
    """軸まわりの回転行列。axis は正規化されていなくてよい。"""
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError("回転軸の長さが 0")
    a = a / n
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    x, y, z = a
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) * c + K * s + np.outer(a, a) * (1.0 - c)


def frame_from_z(zdir, up_hint=(0.0, 0.0, 1.0)):
    """ローカル +Z を `zdir` に向ける回転行列を作る。

    骨は「近位→遠位」をローカル +Z に取る規約なので、これで置ける。
    `up_hint` はローカル +X をどちら寄りにするかの手がかり。
    """
    z = np.asarray(zdir, dtype=np.float64)
    n = np.linalg.norm(z)
    if n < 1e-12:
        raise ValueError("向きベクトルの長さが 0")
    z = z / n
    h = np.asarray(up_hint, dtype=np.float64)
    if abs(float(np.dot(h, z))) > 0.999:
        h = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(h, z))) > 0.999:
            h = np.array([0.0, 1.0, 0.0])
    x = h - z * float(np.dot(h, z))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def compose(R, t):
    return (np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64))


# ── 基底クラス ────────────────────────────────────────────────────────

class Field(object):
    """距離場。

    実装するのは 2 つだけ:
        `_local_dist(P)`  ローカル座標の点群 (N,3) → 距離 (N,)
        `_local_aabb()`   ローカル座標での実体の境界箱 (lo, hi)
    """

    #: 本物の距離（Lipschitz-1 かつ最短距離）を返すか。
    #: False のものは肉厚が安全側に小さく出る。`Offset`/`SmoothUnion` を直接かけない。
    exact = True

    def __init__(self, xform=None, name=""):
        R, t = xform if xform is not None else identity()
        self.R = np.asarray(R, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)
        self.name = name

    # -- 下請け ------------------------------------------------------

    def _local_dist(self, P):
        raise NotImplementedError

    def _local_aabb(self):
        raise NotImplementedError

    # -- 公開 --------------------------------------------------------

    def dist(self, P):
        """世界座標の点群 (N,3) に対する符号つき距離 (N,)。"""
        P = np.asarray(P, dtype=FLOAT)
        flat = P.reshape(-1, 3)
        # 世界 → ローカル は R^T (p - t)。R は直交なので距離は保たれる。
        L = (flat - self.t.astype(FLOAT)) @ self.R.astype(FLOAT)
        return self._local_dist(L).reshape(P.shape[:-1])

    def _world_points_radii(self):
        """「点＋半径」で決まる形なら (点 (N,3), 半径 (N,)) を世界座標で返す。

        球・カプセル・折れ線はこれで**厳密な**境界箱が出せる。
        ローカルの箱を回してから包むと、球1個でも最大 √3 倍に膨らみ、
        間引きの効きがそのぶん落ちる（実測で 12mm → 18.4mm）。
        """
        return None

    def aabb(self):
        """世界座標での実体の境界箱 (lo, hi)。回転を含めて包む。"""
        pr = self._world_points_radii()
        if pr is not None:
            P, r = pr
            W = P @ self.R.T + self.t
            return (np.min(W - r[:, None], axis=0), np.max(W + r[:, None], axis=0))
        lo, hi = self._local_aabb()
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        corners = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                            [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]],
                            [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                            [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]]])
        W = corners @ self.R.T + self.t
        return (W.min(axis=0), W.max(axis=0))

    def placed(self, R=None, t=None):
        """この場を回転・平行移動した**新しい**場を返す（自身は変えない）。"""
        import copy
        out = copy.copy(self)
        Rn = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
        tn = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64)
        out.R = Rn @ self.R
        out.t = Rn @ self.t + tn
        return out


# ── プリミティブ ──────────────────────────────────────────────────────

class Sphere(Field):
    """球。距離場の教科書どおりで、解析解と突合できる基準になる。"""

    def __init__(self, center=(0, 0, 0), radius=1.0, **kw):
        Field.__init__(self, **kw)
        self.c = np.asarray(center, dtype=np.float64)
        self.r = float(radius)

    def _local_dist(self, P):
        d = P - self.c.astype(FLOAT)
        return np.sqrt(np.einsum("ij,ij->i", d, d)) - FLOAT(self.r)

    def _local_aabb(self):
        return (self.c - self.r, self.c + self.r)

    def _world_points_radii(self):
        return (self.c[None, :], np.array([self.r]))


class RoundCone(Field):
    """両端の半径が違うカプセル（丸みのある円錐台）。

    骨幹・肋骨・突起はほぼこれで足りる。r0 == r1 なら普通のカプセル。
    表面は両端の球に外接する円錐面なので、距離は厳密。
    """

    def __init__(self, a, b, r0, r1=None, **kw):
        Field.__init__(self, **kw)
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.r0 = float(r0)
        self.r1 = float(r0 if r1 is None else r1)
        ab = self.b - self.a
        self.L = float(np.linalg.norm(ab))
        if self.L < 1e-9:
            raise ValueError("RoundCone の両端が同じ点")
        dr = self.r1 - self.r0
        if abs(dr) >= self.L:
            raise ValueError(
                "半径差 %.3f が軸長 %.3f 以上。円錐面が定義できない" % (abs(dr), self.L))

    def _local_dist(self, P):
        a = self.a.astype(FLOAT)
        ba = (self.b - self.a).astype(FLOAT)
        L2 = FLOAT(self.L * self.L)
        r0, r1 = FLOAT(self.r0), FLOAT(self.r1)

        pa = P - a
        # 軸方向の正規化パラメータと、軸からの距離
        y = (pa @ ba) / L2                       # 0..1 が線分内
        perp = pa - np.outer(y, ba)
        x = np.sqrt(np.einsum("ij,ij->i", perp, perp))
        y = y * FLOAT(self.L)                    # mm に戻す

        # iq の sdRoundCone。円錐面が軸となす傾き
        L = FLOAT(self.L)
        b = (r0 - r1) / L
        aa = np.sqrt(np.maximum(FLOAT(0.0), FLOAT(1.0) - b * b))
        k = x * (-b) + y * aa

        out = np.empty_like(x)
        m_low = k < FLOAT(0.0)
        m_high = k > aa * L
        m_mid = ~(m_low | m_high)
        out[m_low] = np.sqrt(x[m_low] ** 2 + y[m_low] ** 2) - r0
        out[m_high] = np.sqrt(x[m_high] ** 2 + (y[m_high] - L) ** 2) - r1
        out[m_mid] = x[m_mid] * aa + y[m_mid] * b - r0
        return out

    def _local_aabb(self):
        lo = np.minimum(self.a - self.r0, self.b - self.r1)
        hi = np.maximum(self.a + self.r0, self.b + self.r1)
        return (lo, hi)

    def _world_points_radii(self):
        return (np.stack([self.a, self.b]), np.array([self.r0, self.r1]))


class Box(Field):
    """角を丸めた直方体。ローカル原点中心、半寸法 half。

    椎体・板の芯・台座に使う。round_r を入れると角が丸くなり、
    距離場としての性質も良くなる（尖った角は印刷でも出ない）。
    """

    def __init__(self, half, round_r=0.0, **kw):
        Field.__init__(self, **kw)
        self.h = np.asarray(half, dtype=np.float64) - float(round_r)
        if np.any(self.h < 0):
            raise ValueError("round_r が半寸法より大きい")
        self.rr = float(round_r)

    def _local_dist(self, P):
        q = np.abs(P) - self.h.astype(FLOAT)
        outside = np.linalg.norm(np.maximum(q, FLOAT(0.0)), axis=1)
        inside = np.minimum(q.max(axis=1), FLOAT(0.0))
        return outside + inside - FLOAT(self.rr)

    def _local_aabb(self):
        e = self.h + self.rr
        return (-e, e)


class Ellipsoid(Field):
    """楕円体。関節端・頭蓋の丸みに使う。

    楕円体の厳密な距離は閉じた式で書けない。よく使われる iq の近似式
    ``k0*(k0-1)/k1`` を試したところ、10:6:4 の楕円体で **Lipschitz を
    0.55mm 超過し、勾配の誤差が 1.72、勾配を下っても表面から 3.3mm ずれた**
    （`checks/test_sdf.py` の実測）。これでは肉厚が嘘になるので使わない。

    代わりに、**必ず真の距離以下になる Lipschitz-1 の下界**を使う:

        φ = (|p/r| - 1) × min(r)

    - |∇φ| = min(r)/r_i ≤ 1 なので Lipschitz-1（＝肉厚は安全側にしか出ない）
    - φ = 0 は |p/r| = 1、つまり**表面はまったく厳密**
    - 中心での値 -min(r) も厳密（中心から表面までの最短距離そのもの）
    - 球（3半径が等しい）なら |p| - r になって完全に厳密

    真の距離より小さめに出るので、`Offset` や `SmoothUnion` を直接かけると
    意図より痩せる。関節端の丸みのように「和の材料」として使うこと。
    """

    exact = False

    def __init__(self, radii, **kw):
        Field.__init__(self, **kw)
        self.r = np.asarray(radii, dtype=np.float64)
        if np.any(self.r <= 0):
            raise ValueError("楕円体の半径が 0 以下")

    def _local_dist(self, P):
        r = self.r.astype(FLOAT)
        k0 = np.linalg.norm(P / r, axis=1)
        return (k0 - FLOAT(1.0)) * FLOAT(self.r.min())

    def _local_aabb(self):
        return (-self.r, self.r)


def round_polygon(poly, r, arc_pts=4):
    """多角形の角を半径 r の円弧で落とす。

    **骨に刃物のような角は無い。** それに、尖った角はノズル幅より細い
    スリバーになるので印刷でも出ない。実際、尖らせたままだと角の先端で
    1ボクセルだけ孤立した破片ができ、連結性の検査に引っかかった
    （寛骨の坐骨結節と、下顎の筋突起）。

    凸の角は内側へ、凹（反転）の角は外側へ丸まる。どちらも
    「両辺の向きの和」が指す側に円の中心を置けば同じ式で書ける。
    隣の辺が短くて半径が入らない角は、その角だけ半径を詰める。
    """
    P = np.asarray(poly, dtype=np.float64)
    m = len(P)
    out = []
    for i in range(m):
        v = P[i]
        p = P[(i - 1) % m]
        q = P[(i + 1) % m]
        e1 = p - v
        e2 = q - v
        l1 = np.linalg.norm(e1)
        l2 = np.linalg.norm(e2)
        if l1 < 1e-9 or l2 < 1e-9:
            out.append(v)
            continue
        e1 /= l1
        e2 /= l2
        cosang = float(np.clip(np.dot(e1, e2), -1.0, 1.0))
        ang = math.acos(cosang)                    # 角の開き
        if ang < 1e-3 or abs(ang - math.pi) < 1e-3:
            out.append(v)                          # まっすぐ。丸めない
            continue
        half = ang * 0.5
        rr = float(r)
        t = rr / math.tan(half)                    # 角から接点までの距離
        lim = 0.45 * min(l1, l2)
        if t > lim:                                # 辺が短い角は半径を詰める
            t = lim
            rr = t * math.tan(half)
        bis = e1 + e2
        nb = np.linalg.norm(bis)
        if nb < 1e-9:
            out.append(v)
            continue
        bis /= nb
        c = v + bis * (rr / math.sin(half))
        a1 = v + e1 * t - c
        a2 = v + e2 * t - c
        th1 = math.atan2(a1[1], a1[0])
        th2 = math.atan2(a2[1], a2[0])
        d = th2 - th1
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        for k in range(arc_pts):
            th = th1 + d * (k / float(arc_pts - 1))
            out.append(c + rr * np.array([math.cos(th), math.sin(th)]))
    return np.asarray(out)


class Plate(Field):
    """多角形の輪郭を厚み方向に押し出した板。

    **これが今回の作り直しの核心。** 既存版は断面が常に原点中心・左右対称の
    超楕円だったので、肩甲骨のような「非対称な三角形の平板」が原理的に
    作れなかった。輪郭を多角形で書けば、その制約が消える。

    ローカル座標では XY 平面に輪郭、Z が厚み方向。輪郭は凹んでいてよい。

    thickness:
        数値なら一定厚み。
        呼べるものなら `thickness(x, y) -> 厚み` で、場所ごとに変えられる。
        **可変厚みにすると距離場の厳密さが崩れる**（勾配が 1 を超える）。
        崩れ幅は厚みの変化率でおよそ決まるので、緩やかに変えること。
    """

    def __init__(self, polygon, thickness, corner_r=0.0, **kw):
        Field.__init__(self, **kw)
        poly = np.asarray(polygon, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
            raise ValueError("polygon は (M,2) で M>=3")
        if np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]
        if corner_r > 0:
            poly = round_polygon(poly, corner_r)
        self.corner_r = float(corner_r)
        self.poly = poly
        self.thickness = thickness
        if callable(thickness):
            self._tmax = None       # aabb 用に後で走査する
        else:
            if float(thickness) <= 0:
                raise ValueError("厚みが 0 以下")
            self._tmax = float(thickness)

    # -- 2D 多角形の符号つき距離（iq の sdPolygon をベクトル化） --------

    def _poly_dist(self, xy):
        P = self.poly.astype(FLOAT)
        M = len(P)
        px, py = xy[:, 0], xy[:, 1]
        d = (px - P[0, 0]) ** 2 + (py - P[0, 1]) ** 2
        sign = np.ones_like(px)
        for i in range(M):
            j = (i - 1) % M
            ex, ey = P[j, 0] - P[i, 0], P[j, 1] - P[i, 1]
            wx, wy = px - P[i, 0], py - P[i, 1]
            ee = ex * ex + ey * ey
            tt = np.clip((wx * ex + wy * ey) / ee, FLOAT(0.0), FLOAT(1.0))
            bx, by = wx - ex * tt, wy - ey * tt
            np.minimum(d, bx * bx + by * by, out=d)
            # 交差数による内外判定
            c1 = py >= P[i, 1]
            c2 = py < P[j, 1]
            c3 = (ex * wy) > (ey * wx)
            flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
            sign[flip] *= FLOAT(-1.0)
        return sign * np.sqrt(d)

    def _local_dist(self, P):
        d2 = self._poly_dist(P[:, :2])
        if callable(self.thickness):
            h = np.asarray(self.thickness(P[:, 0], P[:, 1]),
                           dtype=FLOAT) * FLOAT(0.5)
        else:
            h = FLOAT(self.thickness * 0.5)
        dz = np.abs(P[:, 2]) - h
        # 押し出しの厳密な距離（iq の opExtrusion）
        ax = np.maximum(d2, FLOAT(0.0))
        az = np.maximum(dz, FLOAT(0.0))
        return np.minimum(np.maximum(d2, dz), FLOAT(0.0)) + np.sqrt(ax * ax + az * az)

    def _local_aabb(self):
        lo2 = self.poly.min(axis=0)
        hi2 = self.poly.max(axis=0)
        if self._tmax is None:
            # 輪郭の格子点で厚みを走査して最大値を取る
            gx = np.linspace(lo2[0], hi2[0], 24).astype(FLOAT)
            gy = np.linspace(lo2[1], hi2[1], 24).astype(FLOAT)
            X, Y = np.meshgrid(gx, gy, indexing="ij")
            t = np.asarray(self.thickness(X.ravel(), Y.ravel()), dtype=FLOAT)
            self._tmax = float(np.max(t))
        h = self._tmax * 0.5
        return (np.array([lo2[0], lo2[1], -h]), np.array([hi2[0], hi2[1], h]))


class Chain(Field):
    """半径の変わる折れ線（RoundCone の連なり）。

    肋骨・肋軟骨・尾椎のような「曲がりながら細くなるもの」に使う。
    軸が `line` と `arc` しか書けなかった既存版の制約はここには無い。
    """

    def __init__(self, points, radii, **kw):
        Field.__init__(self, **kw)
        P = np.asarray(points, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2:
            raise ValueError("points は (N,3) で N>=2")
        r = np.asarray(radii, dtype=np.float64)
        if r.ndim == 0:
            r = np.full(len(P), float(r))
        if len(r) != len(P):
            raise ValueError("radii の数が points と合わない")
        if np.any(r <= 0):
            raise ValueError("半径が 0 以下")
        self.P, self.r = P, r
        self.segs = []
        for i in range(len(P) - 1):
            L = float(np.linalg.norm(P[i + 1] - P[i]))
            dr = abs(r[i + 1] - r[i])
            if L < 1e-9:
                continue
            if dr >= L:
                # 半径差が大きすぎて円錐面が定義できない。球2つで代用する。
                self.segs.append(Sphere(P[i], r[i]))
                self.segs.append(Sphere(P[i + 1], r[i + 1]))
            else:
                self.segs.append(RoundCone(P[i], P[i + 1], r[i], r[i + 1]))
        if not self.segs:
            raise ValueError("有効な区間が無い")

    def _local_dist(self, P):
        out = self.segs[0]._local_dist(P)
        for s in self.segs[1:]:
            np.minimum(out, s._local_dist(P), out=out)
        return out

    def _local_aabb(self):
        lo = (self.P - self.r[:, None]).min(axis=0)
        hi = (self.P + self.r[:, None]).max(axis=0)
        return (lo, hi)

    def _world_points_radii(self):
        return (self.P, self.r)
