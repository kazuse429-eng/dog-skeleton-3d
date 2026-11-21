# -*- coding: utf-8 -*-
"""中型犬（体高 約380mm）の全身骨格を、骨ごとの距離場として定義する。

座標系（mm）: +X = 頭側, +Y = 左, +Z = 上, Z=0 = 地面。
左右のある骨は **左だけ** 作る。右はビューア側で Y を反転して作る。

形は解剖書の模式図を立体にした程度の精度。位置関係・個数・比率は
一般的な獣医解剖学の記載（椎式 C7 T13 L7 S3 Cd20、肋骨13対 等）に合わせている。
"""

import math

import numpy as np

from sdf import ops
from sdf.prim import Box, Chain, Ellipsoid, Plate, RoundCone, Sphere


# ─── 小道具 ──────────────────────────────────────────────────────────────

def V(*a):
    return np.array(a, dtype=float)


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def lerp(a, b, t):
    return a + (b - a) * float(t)


def frame(d, up_hint=(0.0, 0.0, 1.0)):
    """進む向き d から (d, up, left) の正規直交枠。"""
    d = unit(d)
    up = np.asarray(up_hint, dtype=float)
    up = unit(up - d * float(np.dot(up, d)))
    left = np.cross(up, d)
    return d, up, left


def plate(poly, th, origin, u, v, corner=0.8, name=""):
    """面 (u, v) に多角形の板を置く。厚みは面の法線方向に ±th/2。"""
    u = unit(u)
    v = np.asarray(v, dtype=float)
    v = unit(v - u * float(np.dot(v, u)))
    R = np.column_stack([u, v, np.cross(u, v)])
    return Plate(poly, th, corner_r=corner, xform=(R, np.asarray(origin, dtype=float)),
                 name=name)


def catmull(ctrl, n=60):
    P = [np.asarray(c, dtype=float) for c in ctrl]
    P = [P[0] + (P[0] - P[1])] + P + [P[-1] + (P[-1] - P[-2])]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0.0, 1.0, n, endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2])
    return np.array(out)


class Path(object):
    """制御点を通る曲線。弧長で位置と接線を引ける。"""

    def __init__(self, ctrl, n=80):
        self.p = catmull(ctrl, n)
        d = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(d)])
        self.length = float(self.s[-1])

    def at(self, s):
        s = min(max(float(s), 0.0), self.length)
        i = int(np.clip(np.searchsorted(self.s, s), 1, len(self.s) - 1))
        t = (s - self.s[i - 1]) / max(1e-9, self.s[i] - self.s[i - 1])
        return self.p[i - 1] + (self.p[i] - self.p[i - 1]) * t

    def tan(self, s):
        return unit(self.at(s + 1.0) - self.at(s - 1.0))


def smooth_chain(ctrl, radii, n=8, name=""):
    """制御点をなめらかに通る、太さの変わる棒。"""
    pts = catmull(ctrl, n)
    r = np.interp(np.linspace(0, 1, len(pts)), np.linspace(0, 1, len(radii)), radii)
    return Chain(pts, r, name=name)


# ─── 骨の入れ物 ──────────────────────────────────────────────────────────

BONES = []


def bone(ident, ja, la, path, field, paired=False, voxel=0.45, note="", layer="bone", color=None,
         alpha=1.0):
    BONES.append(dict(id=ident, ja=ja, la=la, path=list(path), field=field,
                      paired=bool(paired), voxel=float(voxel), note=note,
                      layer=layer, color=color, alpha=float(alpha)))


# ═══════════════════════════════════════════════════════════════════════
# 脊柱
# ═══════════════════════════════════════════════════════════════════════

# 椎体の中心が通る線（正中、頭→尾）。
SPINE = Path([
    (402, 0, 421), (372, 0, 396), (336, 0, 356), (300, 0, 316), (270, 0, 292),
    (200, 0, 290), (120, 0, 298), (40, 0, 308), (-40, 0, 318), (-110, 0, 320),
    (-150, 0, 316), (-180, 0, 306), (-220, 0, 290), (-275, 0, 262),
    (-330, 0, 232), (-385, 0, 212), (-440, 0, 205), (-480, 0, 204),
])

GAP = 1.6   # 椎間（椎間円板のぶん）

SEGMENTS = (
    [("C", i + 1, L) for i, L in enumerate([18, 32, 24, 23, 22, 21, 19])]
    + [("T", i + 1, L) for i, L in enumerate(
        [15, 15, 15.5, 15.5, 16, 16, 16, 16.5, 16.5, 17, 17, 17, 17])]
    + [("L", i + 1, L) for i, L in enumerate([19, 20, 21, 22, 22, 21, 19])]
    + [("S", 1, 34)]
    + [("Cd", i + 1, L) for i, L in enumerate(
        [12, 13, 15, 17, 18, 19, 19, 18, 18, 17, 16, 15, 14, 13, 12, 10, 9, 8, 7, 6])]
)

VERT = {}   # 名前 -> dict(c=中心, d=尾側向き, up, left, L, h, w)


def _place_spine():
    s = 0.0
    for reg, n, L in SEGMENTS:
        c = SPINE.at(s + L * 0.5)
        d, up, left = frame(SPINE.tan(s + L * 0.5))
        VERT["%s%d" % (reg, n)] = dict(reg=reg, n=n, c=c, d=d, up=up, left=left, L=float(L),
                                       s0=s)
        s += L + GAP


_place_spine()

BODY_HW = {"C": (15.0, 17.0), "T": (13.5, 16.0), "L": (16.0, 21.0), "S": (14.0, 26.0)}

SPINE_H = {
    "C": {1: 0, 2: 18, 3: 3, 4: 3, 5: 4, 6: 7, 7: 16},
    "T": {1: 60, 2: 64, 3: 64, 4: 60, 5: 55, 6: 49, 7: 43, 8: 37, 9: 31, 10: 26,
          11: 22, 12: 20, 13: 20},
    "L": {1: 21, 2: 23, 3: 24, 4: 24, 5: 23, 6: 19, 7: 14},
}


def _spine_lean(reg, n):
    """棘突起の傾き（度）。正 = 先端が尾側。犬の対傾椎は第11胸椎。"""
    if reg == "T":
        if n <= 10:
            return lerp(28.0, 14.0, (n - 1) / 9.0)
        if n == 11:
            return 0.0
        return -14.0
    if reg == "L":
        return -18.0
    if reg == "C":
        return 8.0
    return 0.0


def _taper_plate(length, base_w, tip_w, lean_deg, t_base, t_tip, tip_round=True, n_arc=6):
    """根元が広く・厚く、先へ細く・薄くなる突起の輪郭と厚み関数。

    面内: x = 根元→先端, y = 前後。先端は半円で丸める（角ばった板に見えないように）。
    """
    lr = math.tan(math.radians(lean_deg))
    tip_c = (length - tip_w * 0.5, (length - tip_w * 0.5) * lr)
    poly = [(0.0, -base_w * 0.5), (length * 0.35, -base_w * 0.36 + length * 0.35 * lr)]
    if tip_round:
        for k in range(n_arc + 1):
            a = -math.pi / 2 + math.pi * k / n_arc
            poly.append((tip_c[0] + math.cos(a) * tip_w * 0.5, tip_c[1] + math.sin(a) * tip_w * 0.5))
    else:
        poly += [(length, tip_c[1] - tip_w * 0.5), (length, tip_c[1] + tip_w * 0.5)]
    poly += [(length * 0.35, base_w * 0.36 + length * 0.35 * lr), (0.0, base_w * 0.5)]

    def thick(x, y):
        t = np.clip(np.asarray(x) / length, 0.0, 1.0)
        return t_base + (t_tip - t_base) * t ** 0.7

    return poly, thick


def _tplate(poly, thick, origin, u, v, name=""):
    u = unit(u)
    v = np.asarray(v, dtype=float)
    v = unit(v - u * float(np.dot(v, u)))
    R = np.column_stack([u, v, np.cross(u, v)])
    return Plate(poly, thick, corner_r=0.0, xform=(R, np.asarray(origin, dtype=float)), name=name)


CANAL = {"C": 6.2, "T": 5.0, "L": 5.6, "S": 4.2}


def vertebra(key):
    """椎骨1個。

    最初の版は «角を丸めた箱を積んだだけ» で、棘突起は同じ厚みの平板、
    椎弓も箱、つなぎ目はただ重ねただけだったので、標本に見えなかった。
    ここでは
      - 椎弓は «ブロックから椎孔をくり抜いた輪»
      - 棘突起・横突起は «根元が厚く先が薄く、先端が丸い» テーパー板
      - 前後の関節突起を付ける
      - 部品は滑らかな和（k 小さめ）でつなぐ
    """
    v = VERT[key]
    reg, n, c, d, up, left, L = v["reg"], v["n"], v["c"], v["d"], v["up"], v["left"], v["L"]
    R = np.column_stack([left, up, d])

    if reg == "Cd":
        k = (n - 1) / 19.0
        r = lerp(6.0, 2.0, k ** 0.8)
        a = c - d * (L * 0.5)
        b = c + d * (L * 0.5)
        parts = [Chain([a, c, b], [r * 1.12, r * 0.80, r * 1.12])]
        if n <= 6:
            tl = lerp(9.0, 3.0, (n - 1) / 5.0)
            for sg in (1, -1):
                parts.append(RoundCone(c - d * (L * 0.15), c + left * (sg * (r + tl))
                                       + d * (L * 0.05), 2.0, 1.2))
            if n <= 4:
                parts.append(RoundCone(c + up * (r * 0.9) - d * (L * 0.3),
                                       c + up * (r * 1.7), 1.8, 1.1))
                for sg in (1, -1):   # 小さな前関節突起
                    parts.append(Sphere(c + up * (r * 0.9) + left * (sg * r * 0.6) - d * (L * 0.48), 1.4))
        if len(parts) == 1:
            return parts[0]
        return ops.SmoothUnion(parts, k=1.2, name=key)

    h, w = BODY_HW[reg]
    rc = CANAL[reg]
    parts = []

    # ── 環椎（C1）: 椎体が無い輪＋大きな翼 ──────────────────────
    if reg == "C" and n == 1:
        ring_c = c + up * 2.0
        ring = ops.Difference([
            Ellipsoid((rc + 5.0, rc + 5.0, L * 0.45), xform=(R, ring_c)),
            Chain([ring_c - d * L, ring_c + d * L], [rc + 0.6, rc + 0.6]),
        ])
        parts.append(ring)
        for sg in (1, -1):
            poly, th = _taper_plate(22.0, L * 1.05, L * 0.80, 0.0, 4.0, 2.2, n_arc=8)
            parts.append(_tplate(poly, th, ring_c + left * (sg * (rc + 2.5)) - up * 1.0,
                                 unit(left * sg - up * 0.25), d))
        parts.append(Ellipsoid((3.0, 2.0, 3.0), xform=(R, ring_c + up * (rc + 4.6))))  # 背側結節
        return ops.SmoothUnion(parts, k=2.0, name=key)

    # ── 椎体 ───────────────────────────────────────────────
    # 端が平らな円柱（楕円断面）。角を丸めた箱だと短い椎体が «まんじゅう» になった。
    # 長い楕円体を椎体の長さの板で切り、0.8 だけ太らせて縁を丸める。
    body = ops.Offset(ops.Intersection([
        Ellipsoid((w * 0.5 - 0.8, h * 0.5 - 0.8, L * 1.1), xform=(R, c)),
        Box((w, h, L * 0.5 - 0.8), xform=(R, c)),
    ]), 0.8)
    parts.append(body)
    if reg == "C" and n >= 3:
        # 腹側稜（後ろへ向かって高くなる）
        parts.append(RoundCone(c - up * (h * 0.40) - d * (L * 0.30),
                               c - up * (h * 0.62) + d * (L * 0.30), 2.0, 2.8))

    # ── 椎弓: ブロックから椎孔をくり抜く ─────────────────────────
    canal_c = c + up * (h * 0.5 + rc * 0.75)
    arch_len = L * (0.36 if reg == "C" else 0.33)
    # 椎弓: 丸い輪（楕円柱をくり抜く）。箱だと四角い窓枠に見えた。
    arch = ops.Difference([
        ops.Offset(ops.Intersection([
            Ellipsoid((rc + 2.0, rc + 2.2, L * 2.0), xform=(R, canal_c + up * 0.8)),
            Box((rc + 4.0, rc + 4.0, arch_len - 0.8), xform=(R, canal_c)),
        ]), 0.8),
        Chain([canal_c - d * L, canal_c + d * L], [rc, rc]),
    ])
    parts.append(arch)
    top = h * 0.5 + rc * 0.75 + rc + 3.4               # 椎弓の上面（中心からの高さ）

    # ── 関節突起（前: 上向きの関節面、後: 下向き） ──────────────────
    ap_lat = rc + (3.6 if reg == "C" else 2.0)
    ap_r = (3.6, 2.6, 4.2) if reg == "C" else (2.4, 2.0, 3.0)
    for sg in (1, -1):
        parts.append(Ellipsoid(ap_r, xform=(R, c + left * (sg * ap_lat) + up * (top - 2.0)
                                            - d * (L * 0.46))))
        parts.append(Ellipsoid(ap_r, xform=(R, c + left * (sg * (ap_lat - 1.0)) + up * (top - 1.0)
                                            + d * (L * 0.46))))
        if reg == "L":
            # 乳頭突起（前関節突起の上の小さなこぶ）
            parts.append(Sphere(c + left * (sg * ap_lat) + up * (top + 1.0) - d * (L * 0.42), 1.6))

    # ── 棘突起 ───────────────────────────────────────────────
    base = c + up * (top - 1.5)
    if reg == "C" and n == 2:
        # 軸椎: 前へ大きく張り出す稜状の棘突起（板の輪郭を前後に長く）
        poly = [(0, -L * 0.55), (8, -L * 0.85), (15, -L * 0.75), (19, -L * 0.30),
                (20, L * 0.15), (16, L * 0.40), (8, L * 0.48), (0, L * 0.45)]
        parts.append(_tplate(poly, lambda x, y: 3.6 - 1.6 * np.clip(np.asarray(x) / 20.0, 0, 1),
                             base, up, d))
        parts.append(Chain([c - d * (L * 0.45), c - d * (L * 0.5 + 6.0), c - d * (L * 0.5 + 10.0)],
                           [4.6, 3.8, 3.0]))                     # 歯突起
    elif reg == "S":
        poly = [(0, -L * 0.45), (7, -L * 0.40), (9, -L * 0.1), (8, L * 0.25), (0, L * 0.45)]
        parts.append(_tplate(poly, lambda x, y: 3.0 - 1.0 * np.clip(np.asarray(x) / 9.0, 0, 1),
                             base, up, d))
    else:
        sh = SPINE_H[reg][n]
        if sh > 2:
            lean = _spine_lean(reg, n)
            if reg == "T":
                poly, th = _taper_plate(sh, L * 0.66, max(5.5, L * 0.38), lean, 3.6, 2.0)
            elif reg == "L":
                poly, th = _taper_plate(sh, L * 0.70, L * 0.55, lean, 3.6, 2.4)
            else:
                poly, th = _taper_plate(sh + 2, L * 0.45, L * 0.30, lean, 3.0, 2.0)
            parts.append(_tplate(poly, th, base, up, d))

    # ── 横突起 ───────────────────────────────────────────────
    if reg == "C" and n >= 3:
        for sg in (1, -1):
            a = c + left * (sg * w * 0.42) - up * (h * 0.05)
            out = unit(left * sg * 0.75 - up * 0.65)
            poly, th = _taper_plate(10.0 if n < 6 else 9.0, L * 0.80, L * 0.55, 0.0, 3.2, 2.0)
            parts.append(_tplate(poly, th, a, out, d))
            if n == 6:   # 第6頸椎の腹側板
                poly, th = _taper_plate(13.0, L * 0.85, L * 0.75, 0.0, 2.6, 1.6, n_arc=8)
                parts.append(_tplate(poly, th, a, unit(left * sg * 0.3 - up), d))
    elif reg == "T":
        for sg in (1, -1):
            a = c + left * (sg * (rc + 1.0)) + up * (top - 3.0)
            b = a + left * (sg * 7.5) + up * 1.5
            parts.append(RoundCone(a, b, 2.8, 2.3))
            parts.append(Sphere(b, 2.7))                         # 横突起の肋骨窩
    elif reg == "L":
        tl = [16, 20, 24, 27, 28, 27, 20][n - 1]
        for sg in (1, -1):
            a = c + left * (sg * w * 0.30)
            out = unit(left * sg - up * 0.22 - d * 0.35)        # 外・前・やや下へ
            poly, th = _taper_plate(tl, L * 0.62, L * 0.45, -12.0, 2.8, 1.6, n_arc=6)
            parts.append(_tplate(poly, th, a, out, d))
    elif reg == "S":
        for sg in (1, -1):
            a = c + left * (sg * w * 0.40) - d * (L * 0.18)
            poly, th = _taper_plate(15.0, L * 0.85, L * 0.55, -10.0, 9.0, 5.0, n_arc=8)
            parts.append(_tplate(poly, th, a, left * sg, d))     # 仙骨翼
        parts.append(Box((w * 0.32, h * 0.40, L * 0.2), round_r=3.0,
                         xform=(R, c + d * (L * 0.45))))
    return ops.SmoothUnion(parts, k=2.2, name=key)


_C_NOTE = {
    1: "頭蓋と関節する。椎体を欠く輪状の骨で、左右に大きな環椎翼が張り出す。翼には翼孔・椎骨孔がある。",
    2: "前方へ歯突起が伸び、環椎の中で回転軸になる。棘突起は前後に長い板状（犬では前方へ大きく突出）。",
    6: "横突起の腹側板（板状の張り出し）が目立つのが特徴。",
    7: "横突孔を欠く。肋骨窩（第1肋骨頭との関節面）を持つ。",
}


def build_spine():
    for i in range(1, 8):
        name = {1: "環椎", 2: "軸椎"}.get(i, "第%d頸椎" % i)
        la = {1: "Atlas (C1)", 2: "Axis (C2)"}.get(i, "Vertebra cervicalis %d (C%d)" % (i, i))
        bone("C%d" % i, name, la, ["脊柱", "頸椎（7個）"], vertebra("C%d" % i),
             note=_C_NOTE.get(i, "頸椎は7個。横突孔を椎骨動脈が通る。"))
    for i in range(1, 14):
        note = "胸椎は13個。肋骨と関節する（肋骨窩）。"
        if i == 11:
            note = "対傾椎：棘突起がほぼ垂直で、これより前は後ろへ、後ろは前へ傾く（犬では第11胸椎）。"
        elif i <= 3:
            note = "棘突起が最も高い部分（き甲）。胸椎は13個。"
        bone("T%d" % i, "第%d胸椎" % i, "Vertebra thoracica %d (T%d)" % (i, i),
             ["脊柱", "胸椎（13個）"], vertebra("T%d" % i), note=note)
    for i in range(1, 8):
        bone("L%d" % i, "第%d腰椎" % i, "Vertebra lumbalis %d (L%d)" % (i, i),
             ["脊柱", "腰椎（7個）"], vertebra("L%d" % i),
             note="腰椎は7個。横突起は長く、頭側かつ腹側へ向かう。")
    bone("S1", "仙骨", "Os sacrum", ["脊柱"], vertebra("S1"),
         note="3個の仙椎が癒合した骨。仙骨翼で腸骨と関節する（仙腸関節）。")
    for i in range(1, 21):
        bone("Cd%d" % i, "第%d尾椎" % i, "Vertebra caudalis %d (Cd%d)" % (i, i),
             ["脊柱", "尾椎（約20個）"], vertebra("Cd%d" % i), voxel=0.35,
             note="尾椎は約20〜23個（個体差あり）。後方ほど突起を失い、棒状になる。")


# ═══════════════════════════════════════════════════════════════════════
# 胸郭
# ═══════════════════════════════════════════════════════════════════════

_T1 = VERT["T1"]["c"]
MANUBRIUM = V(_T1[0] + 38, 0, 214)
XIPHOID = V(VERT["T11"]["c"][0] - 6, 0, 170)
N_STERNEBRAE = 8


def sternum_point(t):
    return lerp(MANUBRIUM, XIPHOID, t)


def sternal_attach(n):
    """肋骨 n（1〜9）の肋軟骨が胸骨に付く点。"""
    if n == 1:
        t = 0.05
    elif n <= 8:
        t = (n - 1) / float(N_STERNEBRAE)
    else:
        t = 7.4 / N_STERNEBRAE
    p = sternum_point(t)
    return p + V(0, 5.0, 0)


RIB_W = {1: 40, 2: 52, 3: 62, 4: 70, 5: 76, 6: 80, 7: 83, 8: 84, 9: 84, 10: 83,
         11: 81, 12: 78, 13: 74}

RIB_END = {}   # 肋骨の腹側端（肋骨肋軟骨結合）


def rib_end(n):
    if n in RIB_END:
        return RIB_END[n]
    if n <= 9:
        S = sternal_attach(n)
        e = V(S[0] - (6 + 4.2 * n), RIB_W[n] * (0.50 + 0.012 * n), S[2] + 10 + 3.6 * n)
    else:
        T = VERT["T%d" % n]["c"]
        e = V(T[0] - 70 - 4 * (n - 10), RIB_W[n] * 0.78, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                          226, 240, 252, 262][n])
    RIB_END[n] = e
    return e


def rib(n):
    T = VERT["T%d" % n]
    c, d, up, L = T["c"], T["d"], T["up"], T["L"]
    head = c - d * (L * 0.45) + V(0, 7.0, 0) + up * 2.0
    tub = c + up * (T["L"] * 0 + 13.0) + V(0, 15.0, 0)
    e = rib_end(n)
    W = RIB_W[n]
    zv = c[2]
    Dz = zv - e[2]
    Lx = c[0] - e[0]
    ctrl = [head, tub,
            V(c[0] - Lx * 0.20, W * 0.80, zv - Dz * 0.12),
            V(c[0] - Lx * 0.48, W * 0.99, zv - Dz * 0.42),
            V(c[0] - Lx * 0.78, W * 0.86, zv - Dz * 0.76),
            e]
    r = 3.4 if n == 1 else lerp(3.0, 2.4, (n - 2) / 11.0)
    return smooth_chain(ctrl, [r * 1.25, r * 1.1, r, r, r * 0.95, r * 0.9], n=4,
                        name="rib%d" % n)


def costal_cartilage(n):
    e = rib_end(n)
    r = 2.1
    if n <= 9:
        S = sternal_attach(n)
        mid = lerp(e, S, 0.5) + V(-4, 4, -4)
        return smooth_chain([e, mid, S], [r, r, r * 0.95], n=8)
    if n <= 12:
        # 肋骨弓: 前の肋軟骨に付く
        prev_e = rib_end(n - 1)
        if n - 1 <= 9:
            tgt = lerp(prev_e, sternal_attach(n - 1), 0.45)
        else:
            tgt = lerp(prev_e, rib_end(n - 2), 0.35)
        mid = lerp(e, tgt, 0.5) + V(0, 2, -6)
        return smooth_chain([e, mid, tgt], [r * 0.95, r * 0.9, r * 0.85], n=8)
    # 第13肋（浮肋）: 先端の短い軟骨
    tip = e + V(-6, -4, -12)
    return smooth_chain([e, tip], [r * 0.9, r * 0.6], n=4)


def build_thorax():
    for n in range(1, 14):
        if n <= 9:
            kind = "真肋（胸骨肋）：肋軟骨が直接胸骨に付く。"
        elif n <= 12:
            kind = "仮肋（非胸骨肋）：肋軟骨が前の肋軟骨に付き、肋骨弓をつくる。"
        else:
            kind = "浮肋：肋軟骨がどこにも付かず、先端が遊離する。"
        bone("rib%d" % n, "第%d肋骨" % n, "Costa %d" % n, ["胸郭", "肋骨（13対）"],
             rib(n), paired=True, voxel=0.5,
             note="肋骨は13対（真肋9対・仮肋4対、第13肋は浮肋）。" + kind)
        bone("cc%d" % n, "第%d肋軟骨" % n, "Cartilago costalis %d" % n,
             ["胸郭", "肋軟骨"], costal_cartilage(n), paired=True, voxel=0.35,
             note="肋骨の腹側に続く軟骨。" + kind)

    # 胸骨片
    for i in range(1, N_STERNEBRAE + 1):
        t0 = (i - 1) / float(N_STERNEBRAE)
        t1 = i / float(N_STERNEBRAE)
        a = sternum_point(t0)
        b = sternum_point(t1)
        d, up, left = frame(b - a)
        R = np.column_stack([left, up, d])
        c = (a + b) * 0.5
        L = float(np.linalg.norm(b - a))
        if i == 1:
            f = ops.Union([Box((4.5, 5.0, L * 0.42), round_r=2.5, xform=(R, c)),
                           RoundCone(a + d * 2, a - d * 9 + up * 3, 4.0, 2.5)])
            ja, la, note = "胸骨柄", "Manubrium sterni", "胸骨の先頭の胸骨片。第1肋軟骨が付く。前へ胸骨柄軟骨が続く。"
        elif i == N_STERNEBRAE:
            f = ops.Union([Box((4.0, 3.5, L * 0.40), round_r=2.0, xform=(R, c)),
                           plate([(L * 0.35, -7), (L * 0.35 + 22, -11), (L * 0.35 + 26, 0),
                                  (L * 0.35 + 22, 11), (L * 0.35, 7)], 1.6,
                                 c - d * (L * 0.5), d, left, corner=2.0)])
            ja, la, note = "剣状突起", "Processus xiphoideus", "胸骨の最後の胸骨片。後ろに薄い剣状軟骨が続く。"
        else:
            f = Box((4.5, 4.5, L * 0.40), round_r=2.4, xform=(R, c))
            ja, la, note = "第%d胸骨片" % i, "Sternebra %d" % i, "胸骨は8個の胸骨片からなる。胸骨片の間に肋軟骨が付く。"
        bone("st%d" % i, ja, la, ["胸郭", "胸骨（8個の胸骨片）"], f, voxel=0.35, note=note)


# ═══════════════════════════════════════════════════════════════════════
# 前肢（左）
# ═══════════════════════════════════════════════════════════════════════

_G = V(_T1[0] + 38, 58, 287)      # 肩関節（関節窩）は第1胸椎の前下
FORE = dict(
    scap_top=_G + V(-78, -24, 77),
    glenoid=_G,
    elbow=_G + V(-60, 0, -111),
    olecranon=_G + V(-86, 0, -91),
    carpus_top=_G + V(-50, -6, -217),
    carpus_bot=_G + V(-48, -6, -239),
)


def scapula():
    """肩甲骨。輪郭を側面（x, z）で決めてから、胸壁に沿う傾いた面に置く。

    最初は長軸に直交する向きを板の横軸にしていたので、背側縁が前上がりになり、
    頭側角が背骨より上へ飛び出した。背側縁はほぼ水平で、き甲の棘突起より低い。
    """
    G = FORE["glenoid"]
    # 関節窩からの (前後, 上下)
    outline = [(5, -1), (-2, -6), (-12, -2), (-16, 14), (-34, 40), (-62, 62), (-90, 70),
               (-64, 79), (-40, 79), (-22, 66), (-12, 46), (-4, 16)]

    def to3d(dx, dz):
        t = min(max(dz / 80.0, 0.0), 1.0)
        return V(G[0] + dx, lerp(G[1], G[1] - 22.0, t), G[2] + dz)

    P3 = [to3d(*q) for q in outline]
    p0, p1, p2 = to3d(0, 0), to3d(-40, 79), to3d(-90, 70)
    u = unit(p1 - p0)
    w = unit(np.cross(u, p2 - p0))
    if w[1] < 0:
        w = -w                      # 外側（+Y）向き
    v = np.cross(w, u)
    poly = [(float(np.dot(q - p0, u)), float(np.dot(q - p0, v))) for q in P3]
    GL = float(np.linalg.norm(p1 - p0))

    def thick(x, y):
        t = np.clip(1.0 - np.asarray(x) / GL, 0.0, 1.0)
        return 2.0 + 7.0 * t ** 4

    blade = Plate(poly, thick, corner_r=0.0,
                  xform=(np.column_stack([u, v, w]), p0))
    # 肩甲棘: 外側面を背側縁の中央から肩峰まで走る稜
    s0 = to3d(-58, 76)
    s1 = to3d(-6, 12)
    su = unit(s1 - s0)
    SL = float(np.linalg.norm(s1 - s0))
    spine = plate([(0, 0), (SL * 0.25, 9), (SL * 0.75, 11), (SL, 9), (SL * 1.02, 0)],
                  2.2, s0, su, w, corner=0.0)
    glen_cup = Ellipsoid((9.0, 7.5, 6.0), xform=(np.column_stack([u, v, w]), p0 + u * 2.0))
    tub = Sphere(to3d(8, 6), 4.0)                     # 関節上結節
    return ops.Union([blade, spine, glen_cup, tub], name="scapula")


def humerus():
    g, e = FORE["glenoid"], FORE["elbow"]
    ax = unit(e - g)
    cran = unit(V(1, 0, 0) - ax * ax[0])
    lat = V(0, 1, 0)
    head = Sphere(g + ax * 7.0 - cran * 3.0, 10.0)
    greater = Ellipsoid((7.0, 6.0, 9.0), xform=(np.eye(3), g + cran * 10.0 + lat * 3.0 + ax * 4.0))
    shaft = Chain([g + ax * 12.0 + cran * 3.0, lerp(g, e, 0.45) + cran * 2.0,
                   e - ax * 18.0 + cran * 1.0, e - ax * 6.0],
                  [8.0, 5.6, 6.0, 8.0])
    condyle = Ellipsoid((9.0, 13.0, 8.5),
                        xform=(np.column_stack([cran, lat, np.cross(cran, lat)]), e))
    epi_m = Sphere(e - lat * 12.0 - cran * 3.0 + ax * -3.0, 5.0)
    epi_l = Sphere(e + lat * 12.0 - cran * 3.0 + ax * -3.0, 4.6)
    solid = ops.Union([head, greater, shaft, condyle, epi_m, epi_l])
    # 滑車上孔（犬の上腕骨の特徴）: 遠位端を前後に貫く小孔
    hole = RoundCone(e - ax * 14.0 + cran * 15.0, e - ax * 14.0 - cran * 15.0, 2.6, 2.6)
    return ops.Difference([solid, hole], name="humerus")


def radius_ulna():
    e, ct, ol = FORE["elbow"], FORE["carpus_top"], FORE["olecranon"]
    lat = V(0, 1, 0)
    # 橈骨: 近位は外側・前、遠位は内側・前
    r0 = e + V(5, -2, -6)
    r1 = ct + V(2, -3, 4)
    radius = Chain([r0, lerp(r0, r1, 0.5) + V(3, 0, 0), r1 - V(0, 0, 4), r1],
                   [6.5, 4.6, 6.0, 8.0], name="radius")
    radius = ops.Union([radius, Ellipsoid((7.0, 10.0, 4.0), xform=(np.eye(3), r1))])
    # 尺骨: 肘頭から細くなりながら外側の茎状突起へ
    u0 = ol
    u1 = e + V(-9, 2, -6)
    u2 = ct + V(-5, 7, 6)
    ulna = ops.Union([
        Chain([u0 + V(-4, 0, 6), u0, u1, lerp(u1, u2, 0.5), u2],
              [6.0, 7.0, 6.0, 3.6, 3.4]),
        RoundCone(u1, u1 + V(9, 0, 9), 4.0, 2.8),        # 肘突起
        RoundCone(u2, u2 + V(1, 1, -6), 3.2, 2.4),       # 茎状突起
    ], name="ulna")
    return radius, ulna


def carpals():
    """手根骨7個: 近位列3（橈側中間・尺側・副）＋遠位列4（第1〜4）。"""
    top, bot = FORE["carpus_top"], FORE["carpus_bot"]
    z1 = top[2] - 6.0
    z2 = bot[2] + 5.0
    x = top[0] + 1.0
    out = []
    out.append(("ci", "橈側中間手根骨", "Os carpi intermedioradiale",
                Ellipsoid((8.0, 7.5, 5.0), xform=(np.eye(3), V(x, top[1] - 3.5, z1))),
                "近位列の最大の骨。食肉類では橈側手根骨と中間手根骨が癒合している。"))
    out.append(("cu", "尺側手根骨", "Os carpi ulnare",
                Ellipsoid((6.0, 4.0, 4.8), xform=(np.eye(3), V(x - 1, top[1] + 8.0, z1))),
                "近位列の外側（尺側）の骨。"))
    out.append(("ca", "副手根骨", "Os carpi accessorium",
                RoundCone(V(x - 6, top[1] + 6.5, z1 + 1), V(x - 17, top[1] + 6.5, z1 + 3), 4.2, 3.6),
                "後方（掌側）へ突き出す骨。屈筋の付着部になる。"))
    ys = [-9.5, -3.6, 2.4, 8.6]
    rs = [(3.0, 2.6, 3.4), (4.0, 2.8, 3.4), (4.6, 3.0, 3.6), (5.0, 3.6, 3.6)]
    for i in range(4):
        out.append(("c%d" % (i + 1), "第%d手根骨" % (i + 1), "Os carpale %s" % "I II III IV".split()[i],
                    Ellipsoid(rs[i], xform=(np.eye(3), V(x + 1, top[1] + ys[i], z2))),
                    "遠位列の手根骨。遠位列は第1〜第4の4個。"))
    return out


# 指（前肢）。y は肢軸からの外側オフセット、mc = 中手骨の長さ
FORE_DIGITS = {1: dict(y=-13.5, mc=16, top=46, ph=[10, 0, 8]),
               2: dict(y=-8.0, mc=40, top=44, ph=[17, 10, 11]),
               3: dict(y=-2.6, mc=46, top=44, ph=[20, 12, 12]),
               4: dict(y=2.8, mc=46, top=44, ph=[20, 12, 12]),
               5: dict(y=8.4, mc=40, top=44, ph=[17, 10, 11])}

HIND_DIGITS = {2: dict(y=-8.0, mc=52, ph=[17, 10, 11]),
               3: dict(y=-2.6, mc=58, ph=[20, 12, 12]),
               4: dict(y=2.8, mc=58, ph=[20, 12, 12]),
               5: dict(y=8.4, mc=52, ph=[17, 10, 11])}

ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}


def digit_bones(base_x, base_y, base_z, spec, splay, toe_len_scale, prefix, side_word,
                group_mc, group_ph, word_mc, word_digit, la_mc, num, dewclaw=False):
    """中手（足）骨1本と指（趾）骨。地面へ向けて置く。"""
    y = base_y + spec["y"]
    top = V(base_x, y, base_z)
    mc_len = spec["mc"]
    # 中手骨: やや前下へ
    ang = math.radians(70.0 if not dewclaw else 60.0)
    dirv = V(math.cos(ang), splay * spec["y"] * 0.010, -math.sin(ang))
    dirv = unit(dirv)
    bot = top + dirv * mc_len
    if not dewclaw:
        # 中手骨の遠位端を地面から一定の高さにそろえる
        shift = 11.5 - bot[2]
        top = top + V(0, 0, shift)
        bot = bot + V(0, 0, shift)
    r = 3.0 if num in (3, 4) else (2.6 if not dewclaw else 2.2)
    mc = Chain([top, lerp(top, bot, 0.5), bot], [r * 1.25, r * 0.85, r * 1.15])
    bone("%s_mc%d" % (prefix, num), "第%d%s" % (num, word_mc), "%s %s" % (la_mc, ROMAN[num]),
         group_mc, mc, paired=True, voxel=0.3,
         note=("第1%sは短く、地面に着かない（狼爪）。" % word_mc) if dewclaw else
         "犬は趾行性で、%sは地面から立ち上がる。" % word_mc)
    # 指骨
    p = bot
    names = [("基節骨", "Phalanx proximalis"), ("中節骨", "Phalanx media"),
             ("末節骨", "Phalanx distalis")]
    lens = spec["ph"]
    if dewclaw:
        dirs = [unit(V(0.8, 0.0, -0.6)), None, unit(V(0.6, 0.0, -0.8))]
    else:
        dirs = [unit(V(0.85, splay * spec["y"] * 0.012, -0.45)),
                unit(V(0.92, splay * spec["y"] * 0.010, -0.12)),
                unit(V(0.70, 0.0, -0.72))]
    for k in range(3):
        if lens[k] <= 0:
            continue
        q = p + dirs[k] * lens[k] * toe_len_scale
        rr = r * [0.95, 0.85, 0.75][k]
        if k < 2:
            f = Chain([p + dirs[k] * 1.5, lerp(p, q, 0.5), q], [rr * 1.15, rr * 0.8, rr * 1.0])
        else:
            # 末節骨: 爪の芯（鉤爪突起）と爪稜
            tip = q + V(0, 0, -1.5)
            f = ops.Union([Chain([p + dirs[k] * 1.5, lerp(p, tip, 0.45), tip],
                                 [rr * 1.2, rr * 0.75, 0.6]),
                           Sphere(p + dirs[k] * 2.5, rr * 1.15)])
        word = word_digit
        bone("%s_p%d_%d" % (prefix, num, k + 1), "第%d%s %s" % (num, word, names[k][0]),
             "%s (%s %s)" % (names[k][1], word == "指" and "Digitus manus" or "Digitus pedis",
                             ROMAN[num]),
             group_ph + ["第%d%s" % (num, word)], f, paired=True, voxel=0.25,
             note=("指骨は基節骨・中節骨・末節骨の3個（第1%sは中節骨を欠き2個）。" % word)
             + ("末節骨は爪を支える鉤爪突起を持つ。" if k == 2 else ""))
        p = q


def build_forelimb():
    grp = ["前肢"]
    bone("scapula", "肩甲骨", "Scapula", grp, scapula(), paired=True, voxel=0.45,
         note="外側面に肩甲棘があり、下端が肩峰。犬の鎖骨は退化して痕跡的（腕頭筋の中の小片）。")
    bone("humerus", "上腕骨", "Humerus", grp, humerus(), paired=True, voxel=0.45,
         note="遠位端に滑車上孔（肘頭窩と鈎突窩をつなぐ孔）があるのが犬の特徴。")
    rad, uln = radius_ulna()
    bone("radius", "橈骨", "Radius", grp + ["前腕骨"], rad, paired=True, voxel=0.4,
         note="前腕の頭側（前）の骨。近位は外側、遠位は内側へ交差するように尺骨と並ぶ。犬は回内・回外がある程度できる。")
    bone("ulna", "尺骨", "Ulna", grp + ["前腕骨"], uln, paired=True, voxel=0.4,
         note="近位端の肘頭が肘の突端。肘突起は肘頭窩にはまる（肘突起癒合不全は大型犬の疾患）。")
    for ident, ja, la, f, note in carpals():
        bone("carpal_" + ident, ja, la, grp + ["手根骨（7個）"], f, paired=True, voxel=0.25,
             note=note + " 犬の手根骨は7個。")
    base = FORE["carpus_bot"]
    for num, spec in FORE_DIGITS.items():
        dew = num == 1
        bx = base[0] + (2.0 if not dew else -1.0)
        bz = spec["top"] if dew else base[2] - 3
        digit_bones(bx, base[1], bz, spec, 1.0, 1.0, "fore", "左",
                    grp + ["中手骨（5本）"], grp + ["指骨"], "中手骨", "指",
                    "Os metacarpale", num, dewclaw=dew)


# ═══════════════════════════════════════════════════════════════════════
# 後肢（左）
# ═══════════════════════════════════════════════════════════════════════

def _hind_landmarks():
    S = VERT["S1"]
    sac = S["c"]
    hip = V(sac[0] - 12, 50, sac[2] - 46)
    return dict(
        hip=hip,
        tuber_coxae=V(sac[0] + 52, 52, sac[2] + 6),
        sacro=V(sac[0] + 6, 26, sac[2] + 4),
        tuber_isch=V(hip[0] - 68, 40, hip[2] + 4),
        symph_cr=V(hip[0] - 14, 3, hip[2] - 24),
        symph_cd=V(hip[0] - 60, 3, hip[2] - 22),
        stifle=V(hip[0] + 52, 56, 162),
        hock=V(hip[0] - 40, 54, 92),
    )


HIND = _hind_landmarks()


def pelvis():
    hip, tc, sr = HIND["hip"], HIND["tuber_coxae"], HIND["sacro"]
    ti, sc, sd = HIND["tuber_isch"], HIND["symph_cr"], HIND["symph_cd"]

    # 腸骨: 寛骨臼から前上へ。翼はほぼ矢状面に立つ板。
    u = unit(tc - hip)
    # 面内のもう一方の軸は矢状面内で u に直交する向き（背尾側）。
    # 以前は直交化せずに列に並べていて、回転行列が歪み、板が横倒しの棒になった。
    v = unit(np.cross(u, V(0, 1, 0)))
    L = float(np.linalg.norm(tc - hip))
    il_poly = [(4, -7), (L * 0.45, -10), (L * 0.85, -22), (L * 1.04, -18), (L * 1.10, 2),
               (L * 1.00, 20), (L * 0.70, 17), (L * 0.40, 9), (4, 8)]
    ilium = Plate(il_poly, lambda x, y: 4.5 + 4.5 * np.exp(-(np.asarray(x) / 18.0) ** 2),
                  corner_r=2.5,
                  xform=(np.column_stack([u, v, np.cross(u, v)]), hip))
    # 仙骨と接する面（耳状面）へ寄せる内側の張り出し
    ilium = ops.Union([ilium, RoundCone(lerp(hip, tc, 0.62), sr + V(0, 4, 0), 7.0, 6.0),
                       Sphere(hip, 9.5)])
    acet = Sphere(hip + V(0, 7, 0), 9.0)
    ilium = ops.Difference([ilium, acet], name="ilium")

    # 坐骨: 寛骨臼から後ろへ、坐骨結節で終わる。内側は坐骨板（骨盤底）。
    isch = ops.Union([
        RoundCone(hip + V(-6, 0, -2), ti, 6.5, 6.0),
        Ellipsoid((7.0, 5.0, 6.5), xform=(np.eye(3), ti + V(-2, 3, 0))),   # 坐骨結節
        RoundCone(ti, sd, 5.0, 4.5),                                        # 坐骨弓〜骨盤結合
        RoundCone(lerp(ti, sd, 0.4), lerp(hip, sd, 0.6), 4.0, 4.0),         # 閉鎖孔の後縁
    ])
    isch = ops.Difference([isch, acet], name="ischium")
    # 恥骨: 寛骨臼から内腹側へ、骨盤結合へ。
    pub = ops.Union([RoundCone(hip + V(2, 0, -4), sc, 5.2, 4.5),
                     RoundCone(sc, sd, 4.5, 4.5)])
    pub = ops.Difference([pub, acet], name="pubis")
    return ilium, isch, pub


def femur():
    hip, st = HIND["hip"], HIND["stifle"]
    ax = unit(st - hip)
    cran = unit(V(1, 0, 0) - ax * ax[0])
    lat = V(0, 1, 0)
    head = Sphere(hip, 9.0)
    neck = RoundCone(hip, hip + lat * 9 + ax * 6, 6.0, 7.0)
    troch = Ellipsoid((7.0, 5.5, 8.0), xform=(np.eye(3), hip + lat * 13 + ax * 2 - cran * 2 + V(0, 0, 5)))
    shaft = Chain([hip + lat * 8 + ax * 8, lerp(hip + lat * 6, st, 0.5) + cran * 1.5,
                   st - ax * 14, st - ax * 4], [7.5, 5.4, 6.4, 8.0])
    R = np.column_stack([cran, lat, np.cross(cran, lat)])
    cond_m = Ellipsoid((9.0, 6.0, 9.0), xform=(R, st - lat * 7 - cran * 3))
    cond_l = Ellipsoid((9.0, 6.0, 9.0), xform=(R, st + lat * 7 - cran * 3))
    troch_ridge = Ellipsoid((6.0, 9.0, 10.0), xform=(R, st + cran * 4 + ax * -4))
    return ops.Union([head, neck, troch, shaft, cond_m, cond_l, troch_ridge], name="femur")


def tibia_fibula():
    st, hk = HIND["stifle"], HIND["hock"]
    ax = unit(hk - st)
    cran = unit(V(1, 0, 0) - ax * ax[0])
    lat = V(0, 1, 0)
    t0 = st + ax * 12.0
    R = np.column_stack([cran, lat, np.cross(cran, lat)])
    plateau = Ellipsoid((11.0, 15.0, 6.5), xform=(R, t0 - cran * 2))
    crest = plate([(0, -3), (28, -4), (34, 2), (0, 8)], 3.6, t0 + cran * 4,
                  ax, cran, corner=2.0)
    shaft = Chain([t0, lerp(t0, hk, 0.4), hk - ax * 10, hk - ax * 3],
                  [7.0, 5.0, 5.2, 7.0])
    malleolus = Sphere(hk - lat * 6.0, 4.0)
    tibia = ops.Union([plateau, crest, shaft, malleolus], name="tibia")
    f0 = t0 + lat * 12.0 - cran * 6.0 + ax * 2
    f1 = hk + lat * 8.0 - cran * 2.0
    fibula = ops.Union([Chain([f0, lerp(f0, f1, 0.5), f1], [3.6, 1.8, 3.2]),
                        Sphere(f1, 3.6)], name="fibula")
    patella = Ellipsoid((4.5, 5.5, 8.0),
                        xform=(np.column_stack([lat, cran, np.cross(lat, cran)]),
                               st + cran * 15 + ax * -12))
    fabellae = [Sphere(st - cran * 13 + lat * s * 8 + ax * -10, 3.0) for s in (1, -1)]
    return tibia, fibula, patella, ops.Union(fabellae, name="fabellae")


def tarsals():
    hk = HIND["hock"]
    x, y, z = hk
    out = []
    out.append(("talus", "距骨", "Talus",
                ops.Union([Ellipsoid((8.0, 7.0, 6.0), xform=(np.eye(3), V(x + 1, y - 3, z - 8)))]),
                "脛骨と関節する滑車を持つ。"))
    out.append(("calc", "踵骨", "Calcaneus",
                ops.Union([RoundCone(V(x, y + 5, z - 12), V(x - 16, y + 5, z + 14), 5.0, 6.0),
                           Ellipsoid((5, 4.5, 6), xform=(np.eye(3), V(x - 17, y + 5, z + 16)))]),
                "踵骨隆起が飛節の突端（アキレス腱の付着部）。"))
    out.append(("tc", "中心足根骨", "Os tarsi centrale",
                Ellipsoid((6.0, 6.5, 3.4), xform=(np.eye(3), V(x + 3, y - 3, z - 18))),
                "距骨と遠位列のあいだの骨。"))
    ys = [-9.0, -4.0, 1.5, 7.0]
    rs = [(3.0, 2.4, 3.6), (4.0, 2.8, 3.6), (4.6, 3.0, 3.6), (5.0, 3.6, 6.0)]
    zz = [z - 25, z - 25, z - 25, z - 22]
    for i in range(4):
        out.append(("t%d" % (i + 1), "第%d足根骨" % (i + 1), "Os tarsale %s" % "I II III IV".split()[i],
                    Ellipsoid(rs[i], xform=(np.eye(3), V(x + 3, y + ys[i], zz[i]))),
                    "遠位列の足根骨。第4足根骨は大きく、近位列の高さまで伸びる。" if i == 3 else "遠位列の足根骨。"))
    return out


def build_hindlimb():
    grp = ["後肢"]
    il, isch, pub = pelvis()
    hg = grp + ["寛骨"]
    bone("ilium", "腸骨", "Os ilium", hg, il, paired=True, voxel=0.45,
         note="寛骨の前部。腸骨翼が仙骨翼と関節する（仙腸関節）。寛骨は腸骨・坐骨・恥骨（＋寛骨臼骨）が癒合したもの。")
    bone("ischium", "坐骨", "Os ischii", hg, isch, paired=True, voxel=0.4,
         note="寛骨の後部。後端の坐骨結節は座ったとき体重がかかる部分。")
    bone("pubis", "恥骨", "Os pubis", hg, pub, paired=True, voxel=0.4,
         note="寛骨の腹内側部。左右が骨盤結合で合わさる。坐骨とのあいだに閉鎖孔ができる。")
    bone("femur", "大腿骨", "Femur", grp, femur(), paired=True, voxel=0.45,
         note="全身でいちばん長い骨の一つ。大腿骨頭が寛骨臼にはまる（股関節形成不全の評価部位）。")
    tib, fib, pat, fab = tibia_fibula()
    bone("patella", "膝蓋骨", "Patella", grp + ["種子骨"], pat, paired=True, voxel=0.3,
         note="大腿四頭筋腱の中にある最大の種子骨。滑車溝を上下に滑る（膝蓋骨脱臼は小型犬に多い）。")
    bone("fabella", "腓腹筋種子骨（ファベラ）", "Ossa sesamoidea m. gastrocnemii", grp + ["種子骨"],
         fab, paired=True, voxel=0.3, note="大腿骨顆の後ろにある左右2個の小さな種子骨。")
    bone("tibia", "脛骨", "Tibia", grp + ["下腿骨"], tib, paired=True, voxel=0.45,
         note="下腿の主な骨。前縁に脛骨稜。近位の脛骨高平部は前十字靭帯断裂の手術（TPLO）で重要。")
    bone("fibula", "腓骨", "Fibula", grp + ["下腿骨"], fib, paired=True, voxel=0.3,
         note="脛骨の外側にある細い骨。犬では全長にわたって存在し、遠位端が外側果になる。")
    for ident, ja, la, f, note in tarsals():
        bone("tarsal_" + ident, ja, la, grp + ["足根骨（7個）"], f, paired=True, voxel=0.25,
             note=note + " 犬の足根骨は7個。")
    hk = HIND["hock"]
    for num, spec in HIND_DIGITS.items():
        digit_bones(hk[0] + 3, hk[1], hk[2] - 28, spec, 1.0, 1.0, "hind", "左",
                    grp + ["中足骨"], grp + ["趾骨"], "中足骨", "趾", "Os metatarsale", num)


# ═══════════════════════════════════════════════════════════════════════
# 頭部
# ═══════════════════════════════════════════════════════════════════════

HEAD_C1 = V(414, 0, 428)            # 後頭顆
HEAD_NOSE_DIR = unit(V(math.cos(math.radians(-28)), 0, math.sin(math.radians(-28))))
SKULL_GL = 176.0


def head_frame():
    u = HEAD_NOSE_DIR
    up = unit(V(0, 0, 1) - u * float(u[2]))
    left = np.cross(up, u)
    return HEAD_C1, u, up, left


def H(x, y, z=0.0):
    """頭の局所座標（x=後頭→鼻、y=上、z=左）→ 世界座標。"""
    o, u, up, left = head_frame()
    return o + u * x + up * y + left * z


def skull():
    o, u, up, left = head_frame()
    R = np.column_stack([u, up, left])
    # 本体は «塊（楕円体）を滑らかにつなぐ» で作る。
    # 最初は側面の輪郭×幅の板で作ったが、側面も上面も平らでレンガに見えた。
    # 犬の頭蓋は 丸い脳頭蓋 → 額の段（ストップ） → 細長い鼻面 で、
    # 鼻面は上が細く、歯の並ぶ下側が広い。塊ごとに置けばそれがそのまま出る。
    def E(c, r):
        return Ellipsoid(r, xform=(R, H(*c)))

    lobes = [
        E((44, 18, 0), (42, 26, 29)),        # 脳頭蓋
        E((8, 17, 0), (10, 23, 25)),         # 後頭部（項稜の土台）
        E((40, -5, 0), (34, 11, 21)),        # 頭蓋底
        E((86, 27, 0), (22, 14, 19)),        # 額（前頭洞のふくらみ）
        E((120, 13, 0), (34, 17, 20)),       # 鼻面の根元
        E((152, 6, 0), (25, 13, 15)),        # 鼻面
        E((171, 3, 0), (9, 10, 11)),         # 鼻先（切歯骨）
        E((104, -9, 0), (30, 11, 27)),       # 上顎の歯列（下が広い）
        E((140, -7, 0), (26, 9, 18)),
    ]
    body = ops.SmoothUnion(lobes, k=7.0)
    crest = plate([(2, 34), (10, 45), (48, 50), (74, 42), (56, 38), (12, 30)],
                  2.6, o, u, up, corner=0.0)      # 外矢状稜
    parts = [body, crest]
    for sg in (1, -1):
        # 頬骨弓: 大きく外へ張り出す
        parts.append(smooth_chain([H(26, -4, sg * 26), H(48, 0, sg * 46), H(74, 0, sg * 47),
                                   H(98, -4, sg * 32), H(108, -8, sg * 25)],
                                  [4.0, 4.4, 4.6, 5.2, 6.0], n=6))
        # 鼓室胞
        parts.append(Ellipsoid((11.0, 8.0, 7.5), xform=(R, H(28, -16, sg * 17))))
        # 頬骨突起（眼窩の後ろの縁の名残）
        parts.append(RoundCone(H(80, 34, sg * 18), H(86, 24, sg * 24), 3.0, 2.0))
        # 上顎の犬歯
        parts.append(Chain([H(150, -12, sg * 13), H(147, -30, sg * 14), H(144, -42, sg * 13)],
                           [5.0, 3.6, 1.0]))
        # 頬歯（列のこぶ）。第4前臼歯（裂肉歯）が大きい
        for x, h, r in [(132, 8, 3.0), (122, 9, 3.2), (110, 11, 3.6), (98, 14, 4.6),
                        (86, 9, 4.0), (78, 7, 3.4)]:
            parts.append(RoundCone(H(x, -28, sg * 27 * (1.0 if x < 120 else 0.92)),
                                   H(x, -28 - h, sg * 26), r, r * 0.4))
        # 切歯
        for z in (4, 9, 14):
            parts.append(RoundCone(H(174, -12, sg * z * 0.95), H(176, -22, sg * z), 2.0, 1.0))
    solid = ops.Union(parts)
    holes = []
    for sg in (1, -1):
        holes.append(Sphere(H(90, 16, sg * 31), 15.0))        # 眼窩
    holes.append(Ellipsoid((8.0, 7.0, 7.0), xform=(R, H(181, 4, 0))))  # 骨鼻孔
    return ops.Difference([solid] + holes, name="skull")


def mandible():
    o, u, up, left = head_frame()
    parts = []
    cond0 = (48, -20)          # 顎関節の位置（頭の局所座標）
    for sg in (1, -1):
        prof = [(0, 0), (8, 6), (16, 34), (24, 30), (30, 6), (40, -6),
                (110, -6), (126, -8), (128, -18), (110, -26), (60, -28),
                (24, -28), (6, -24), (-6, -24), (-4, -10)]
        yaw = math.radians(sg * 9.0)
        uu = u * math.cos(yaw) - left * math.sin(yaw)
        ll = np.cross(up, uu)
        org = H(cond0[0], cond0[1], sg * 30)
        R = np.column_stack([uu, up, ll])
        # 角は輪郭の角丸ではなく一様な太らせで丸める（角丸は点が増えて遅い）
        parts.append(ops.Offset(Plate(
            prof, lambda x, y: 4.5 - 2.5 * np.clip(np.asarray(x) / 128.0, 0, 1),
            corner_r=0.0, xform=(R, org)), 1.5))
        # 下顎頭（横長）
        parts.append(RoundCone(org - ll * 5 * sg, org + ll * 7 * sg, 4.0, 4.0))
        # 犬歯
        parts.append(Chain([org + uu * 112 + up * -8, org + uu * 113 + up * 8,
                            org + uu * 108 + up * 20], [5.0, 3.4, 0.9]))
        # 頬歯（第1後臼歯＝裂肉歯が大きい）
        for x, h, r in [(46, 6, 3.0), (56, 8, 3.6), (68, 13, 4.6), (80, 9, 3.6),
                        (90, 8, 3.2), (100, 6, 2.8)]:
            parts.append(RoundCone(org + uu * x + up * -4, org + uu * x + up * (h - 2),
                                   r, r * 0.4))
    # 下顎結合
    parts.append(RoundCone(H(170, -40, 8), H(170, -40, -8), 6.0, 6.0))
    return ops.Union(parts, name="mandible")


def build_head():
    bone("skull", "頭蓋骨", "Cranium", ["頭部"], skull(), voxel=0.65,
         note="脳頭蓋と顔面骨からなる。頬骨弓が大きく張り出すのが食肉類の特徴。眼窩は後方が骨で閉じず、眼窩靭帯で囲まれる。歯式は 3142/3143（永久歯42本）。")
    bone("mandible", "下顎骨", "Mandibula", ["頭部"], mandible(), voxel=0.5,
         note="左右の下顎骨が前端の下顎結合でつながる。後方の筋突起は大きく、側頭窩へ入り込む。")


# ═══════════════════════════════════════════════════════════════════════

def build_other():
    hip = HIND["hip"]
    a = V(hip[0] + 40, 0, 205)
    bone("ospenis", "陰茎骨（雄）", "Os penis", ["その他"],
         ops.Union([Chain([a, a + V(30, 0, 2), a + V(56, 0, 6)], [3.4, 3.0, 1.6])]),
         voxel=0.3, note="雄犬の陰茎の中にある骨。腹側に尿道が通る溝（尿道溝）がある。")


def build_all():
    del BONES[:]
    build_head()
    build_spine()
    build_thorax()
    build_forelimb()
    build_hindlimb()
    build_other()
    import organs                      # 臓器（骨格の位置を使うので後から組む）
    organs.build()
    return BONES
