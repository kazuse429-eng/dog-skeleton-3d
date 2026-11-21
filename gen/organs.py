# -*- coding: utf-8 -*-
"""臓器。骨格（bones.py）の位置を基準に、胸腔・腹腔の中へ置く。

座標は bones.py と同じ（mm、+X 頭側 / +Y 左 / +Z 上）。
臓器は左右非対称なものが多いので、基本は鏡像を使わず1個ずつ作る。

形は模式図の精度。位置関係（心臓は第3〜6肋骨の間で心尖が左腹側、
右腎は左腎より前、脾臓は胃の大弯の左、など）を優先している。
"""

import math

import numpy as np

import bones as B
from bones import V, unit, lerp, catmull
from sdf import ops
from sdf.prim import Box, Chain, Ellipsoid, Plate, RoundCone, Sphere


# ─── 小道具 ──────────────────────────────────────────────────────────────

def half(p0, n):
    """平面（p0 を通り法線 n）の «法線と逆側» を残す半空間。"""
    n = unit(n)
    a = unit(np.cross(n, V(0, 0, 1)) if abs(n[2]) < 0.9 else np.cross(n, V(1, 0, 0)))
    b = np.cross(n, a)
    R = np.column_stack([a, b, n])
    return Box((3000.0, 3000.0, 3000.0), xform=(R, np.asarray(p0, dtype=float) - n * 3000.0))


def ell(c, r, axis=None):
    """楕円体。axis を与えると、その向きを第1軸にする。"""
    if axis is None:
        return Ellipsoid(r, xform=(np.eye(3), V(*c)))
    d, up, left = B.frame(axis)
    return Ellipsoid(r, xform=(np.column_stack([d, up, left]), V(*c)))


def tube(pts, radii, n=6):
    return B.smooth_chain([V(*p) for p in pts], radii, n=n)


def sternum_z(x):
    m, xp = B.MANUBRIUM, B.XIPHOID
    t = (m[0] - x) / (m[0] - xp[0])
    return float(lerp(m[2], xp[2], min(max(t, 0.0), 1.2)))


# 中の臓器を隠してしまう大きな器官は半透明で描く
ALPHA = {"lung_lcr": 0.42, "lung_lcd": 0.42, "lung_rcr": 0.42, "lung_rmid": 0.42,
         "lung_rcd": 0.42, "lung_acc": 0.42, "diaphragm": 0.35}


def organ(ident, ja, la, system, field, color, note="", voxel=0.8, paired=False, sub=None):
    path = ["臓器", system] + ([sub] if sub else [])
    B.bone(ident, ja, la, path, field, paired=paired, voxel=voxel, note=note,
           layer="organ", color=color, alpha=ALPHA.get(ident, 1.0))


# ─── 体腔の形 ─────────────────────────────────────────────────────────────

def thorax_cavity():
    """胸腔の内側。肋骨ごとの幅と、背骨〜胸骨の高さから楕円体をつなぐ。"""
    parts = []
    for n in range(1, 12):
        T = B.VERT["T%d" % n]["c"]
        x = T[0]
        zt = T[2] + 4.0                       # 背側は椎体の横まで
        zb = sternum_z(x) + 7.0
        W = B.RIB_W[n] - 7.0
        parts.append(ell((x, 0, (zt + zb) * 0.5), (16.0, W, (zt - zb) * 0.5)))
    T1 = B.VERT["T1"]["c"]
    parts.insert(0, ell((T1[0] + 22, 0, T1[2] - 34), (18.0, B.RIB_W[1] - 14.0, 30.0)))
    return ops.SmoothUnion(parts, k=14.0)


# 横隔膜のドーム（この楕円体の «内側» が腹腔側）
DOME_C = V(40, 0, 228)
DOME_R = (116.0, 86.0, 74.0)


def dome(offset=0.0):
    r = tuple(max(1.0, x + offset) for x in DOME_R)
    return Ellipsoid(r, xform=(np.eye(3), DOME_C))


def abdomen_cavity():
    """腹腔の内側。前は肋骨弓の内側、後ろは骨盤入口まで。"""
    pts = [(120, 70, 300, 178), (80, 72, 302, 180), (40, 68, 304, 188), (0, 62, 306, 196),
           (-40, 58, 306, 205), (-80, 52, 300, 218), (-115, 40, 296, 232)]
    parts = [ell((x, 0, (zt + zb) * 0.5), (26.0, w, (zt - zb) * 0.5)) for x, w, zt, zb in pts]
    return ops.SmoothUnion(parts, k=16.0)


# ─── 器官 ─────────────────────────────────────────────────────────────────

HEART_BASE = V(226, 0, 256)
HEART_APEX = V(176, 12, 190)


def build():
    thorax = thorax_cavity()
    abd = abdomen_cavity()
    in_dome = dome(-1.5)                     # 腹腔側（横隔膜の内側）

    # ── 神経系 ──────────────────────────────────────────────
    sysn = "神経系"
    o, u, up, left = B.head_frame()
    R = np.column_stack([u, up, left])
    hemis = []
    for sg in (1, -1):
        hemis.append(ops.Intersection([
            Ellipsoid((33.0, 21.0, 24.0), xform=(R, B.H(50, 18, 0))),
            half(B.H(50, 18, sg * 0.8), -left * sg),
        ]))
    cerebrum = ops.SmoothUnion(hemis + [Ellipsoid((9.0, 5.0, 4.0), xform=(R, B.H(86, 14, 5))),
                                         Ellipsoid((9.0, 5.0, 4.0), xform=(R, B.H(86, 14, -5)))], k=2.0)
    organ("cerebrum", "大脳", "Cerebrum", sysn, cerebrum, "#e8b4b8", voxel=0.6,
          note="左右の大脳半球。前端に嗅球がある（犬は嗅覚が発達し嗅球が大きい）。")
    organ("cerebellum", "小脳", "Cerebellum", sysn,
          Ellipsoid((10.0, 12.0, 15.0), xform=(R, B.H(16, 14, 0))), "#d4949c", voxel=0.5,
          note="大脳の後ろ、脳幹の背側。運動の協調を担う。")
    organ("brainstem", "脳幹", "Truncus encephali", sysn,
          Chain([B.H(40, 8, 0), B.H(20, 4, 0), B.H(-4, 2, 0)], [7.0, 6.0, 4.5]), "#c98b8b",
          voxel=0.5, note="中脳・橋・延髄。大後頭孔を通って脊髄へ続く。")
    cord_pts, cord_r = [], []
    for key in ["C%d" % i for i in range(1, 8)] + ["T%d" % i for i in range(1, 14)] + \
            ["L%d" % i for i in range(1, 7)]:
        v = B.VERT[key]
        h = B.BODY_HW[v["reg"]][0]
        rc = B.CANAL[v["reg"]]
        cord_pts.append(v["c"] + v["up"] * (h * 0.5 + rc * 0.75))
        cord_r.append(3.4 if key[0] == "C" else (3.0 if key[0] == "T" else 2.6))
    cord_pts.insert(0, B.H(-4, 2, 0))
    cord_r.insert(0, 4.0)
    cord_r[-1] = 1.4
    organ("spinalcord", "脊髄", "Medulla spinalis", sysn, Chain(cord_pts, cord_r), "#f0d0a0",
          voxel=0.5, note="脊柱管の中を通る。犬では第6〜7腰椎あたりで終わり、その後ろは馬尾。")

    # ── 呼吸器 ──────────────────────────────────────────────
    sysr = "呼吸器"
    neck = []
    for k in range(2, 8):
        v = B.VERT["C%d" % k]
        neck.append(v["c"] - v["up"] * 26.0)
    t1 = B.VERT["T1"]
    neck.append(t1["c"] - t1["up"] * 30.0)
    bif = V(205, 0, 262)
    trachea_pts = neck + [bif]
    c1 = B.VERT["C1"]
    larynx_c = c1["c"] - c1["up"] * 26.0 + c1["d"] * 4.0
    organ("larynx", "喉頭", "Larynx", sysr,
          ops.SmoothUnion([ell(tuple(larynx_c), (15.0, 12.0, 12.0), axis=c1["d"]),
                           RoundCone(larynx_c, neck[0], 10.0, 7.5)], k=3.0),
          "#d8c8b8", voxel=0.6, note="気管の入口。甲状軟骨・輪状軟骨・披裂軟骨などの軟骨でできる。")
    tr = B.smooth_chain(trachea_pts, [7.0] * len(trachea_pts), n=6)
    bronchi = [RoundCone(bif, bif + V(-22, 14, -8), 5.0, 4.0),
               RoundCone(bif, bif + V(-22, -14, -8), 5.0, 4.0)]
    organ("trachea", "気管", "Trachea", sysr, ops.Union([tr] + bronchi), "#dfe6ea", voxel=0.5,
          note="C字形の気管軟骨が連なる管。心底の背側（第4〜5肋間）で左右の主気管支に分かれる。")

    heart_core = ops.SmoothUnion([
        ell(tuple(lerp(HEART_BASE, HEART_APEX, 0.55)), (40.0, 31.0, 29.0), axis=HEART_APEX - HEART_BASE),
        ell(tuple(HEART_BASE + V(-6, 0, -4)), (22.0, 26.0, 18.0)),
    ], k=8.0)
    lung_space = ops.Difference([thorax,
                                 ops.Offset(heart_core, 4.0), dome(3.0),
                                 ops.Offset(B.smooth_chain(trachea_pts[-2:], [9.0, 9.0], n=2), 1.0)])
    # 左肺: 前葉（前部・後部）と後葉
    L = ops.Intersection([lung_space, half(V(0, 3.0, 0), V(0, -1, 0))])
    fis_L_p, fis_L_n = V(170, 0, 250), unit(V(1.0, 0, 0.35))
    organ("lung_lcr", "左肺 前葉", "Lobus cranialis pulmonis sinistri", sysr,
          ops.Intersection([L, half(fis_L_p + fis_L_n * 1.2, -fis_L_n)]), "#f2a7b0", voxel=0.9,
          note="左肺は前葉（前部・後部）と後葉の2葉。前葉の前部は心臓の前まで伸びる。")
    organ("lung_lcd", "左肺 後葉", "Lobus caudalis pulmonis sinistri", sysr,
          ops.Intersection([L, half(fis_L_p - fis_L_n * 1.2, fis_L_n)]), "#ec97a2", voxel=0.9,
          note="左肺でいちばん大きい葉。横隔膜面がドームに沿ってへこむ。")
    # 右肺: 前葉・中葉・後葉・副葉
    Rl = ops.Intersection([lung_space, half(V(0, -3.0, 0), V(0, 1, 0))])
    acc_region = ell((128, -4, 232), (34.0, 16.0, 26.0))
    Rmain = ops.Difference([Rl, ops.Offset(acc_region, 1.5)])
    f1p, f1n = V(222, 0, 250), unit(V(1.0, 0, 0.25))
    f2p, f2n = V(176, 0, 250), unit(V(1.0, 0, 0.45))
    organ("lung_rcr", "右肺 前葉", "Lobus cranialis pulmonis dextri", sysr,
          ops.Intersection([Rmain, half(f1p + f1n * 1.2, -f1n)]),
          "#f2a7b0", voxel=0.9, note="右肺は前葉・中葉・後葉・副葉の4葉（犬）。")
    organ("lung_rmid", "右肺 中葉", "Lobus medius pulmonis dextri", sysr,
          ops.Intersection([Rmain, half(f1p - f1n * 1.2, f1n), half(f2p + f2n * 1.2, -f2n)]),
          "#f6b6be", voxel=0.9, note="右肺の中葉。心臓の右側面に接する（心切痕）。")
    organ("lung_rcd", "右肺 後葉", "Lobus caudalis pulmonis dextri", sysr,
          ops.Intersection([Rmain, half(f2p - f2n * 1.2, f2n)]), "#ec97a2", voxel=0.9,
          note="右肺の後葉。横隔膜に接する。")
    organ("lung_acc", "右肺 副葉", "Lobus accessorius", sysr,
          ops.Intersection([lung_space, acc_region]), "#e2899a", voxel=0.8,
          note="後大静脈を包むように正中付近にある小さな葉。心臓と横隔膜のあいだ。")

    # ── 循環器 ──────────────────────────────────────────────
    sysc = "循環器"
    heart = ops.SmoothUnion([
        heart_core,
        ell(tuple(HEART_BASE + V(4, 16, -2)), (12.0, 8.0, 9.0)),     # 左心耳
        ell(tuple(HEART_BASE + V(10, -14, -4)), (12.0, 8.0, 9.0)),   # 右心耳
    ], k=5.0)
    organ("heart", "心臓", "Cor", sysc, heart, "#b3262e", voxel=0.7,
          note="第3〜6肋骨のあいだ。心底は背側前方、心尖は左腹側後方を向き、胸骨の近くにある。")
    aorta_pts = [HEART_BASE + V(-2, 2, 4), HEART_BASE + V(8, 4, 22), V(215, 6, 284), V(185, 5, 286)]
    for key in ["T7", "T10", "T13", "L2", "L4", "L5"]:
        v = B.VERT[key]
        h = B.BODY_HW[v["reg"]][0]
        aorta_pts.append(v["c"] - v["up"] * (h * 0.5 + 6.0) + V(0, 4, 0))
    aorta = B.smooth_chain(aorta_pts, list(np.linspace(7.0, 4.5, len(aorta_pts))), n=6)
    l6 = B.VERT["L6"]
    iliac = [RoundCone(aorta_pts[-1], l6["c"] - l6["up"] * 16 + V(-30, s * 22, -8), 3.6, 2.8)
             for s in (1, -1)]
    organ("aorta", "大動脈", "Aorta", sysc, ops.Union([aorta] + iliac), "#d9302c", voxel=0.6,
          note="左心室から出て大動脈弓をつくり、胸椎・腰椎の腹側を後ろへ。腰部で左右の外腸骨動脈に分かれる。")
    cvc_pts = [HEART_BASE + V(-14, -8, -10), V(150, -8, 236), V(110, -10, 250), V(72, -14, 270)]
    for key in ["T13", "L2", "L4", "L5"]:
        v = B.VERT[key]
        h = B.BODY_HW[v["reg"]][0]
        cvc_pts.append(v["c"] - v["up"] * (h * 0.5 + 7.0) + V(0, -8, 0))
    organ("cvc", "後大静脈", "Vena cava caudalis", sysc,
          B.smooth_chain(cvc_pts, list(np.linspace(7.0, 5.0, len(cvc_pts))), n=6), "#3f62b5",
          voxel=0.6, note="後半身の血液を右心房へ戻す。横隔膜の大静脈孔を通り、肝臓の背側を走る。")

    # ── 消化器 ──────────────────────────────────────────────
    sysd = "消化器"
    eso = [larynx_c - c1["up"] * -6.0 + c1["d"] * 10.0]
    for k in range(3, 8):
        v = B.VERT["C%d" % k]
        eso.append(v["c"] - v["up"] * 16.0 + V(0, 6, 0))
    eso += [V(240, 4, 272), V(180, 2, 270), V(120, 4, 266), V(80, 8, 264), V(66, 12, 262)]
    organ("esophagus", "食道", "Oesophagus", sysd,
          B.smooth_chain(eso, [5.0] * len(eso), n=6), "#e8a08c", voxel=0.5,
          note="頸部では気管の左背側、胸部では気管・心底の背側を通り、横隔膜の食道裂孔を抜けて胃の噴門へ。")
    stomach_raw = tube([(66, 12, 262), (60, 30, 266), (52, 40, 244), (50, 34, 214),
                        (56, 12, 198), (68, -10, 204), (80, -24, 218)],
                       [11.0, 23.0, 25.0, 22.0, 17.0, 12.0, 7.0], n=6)
    stomach = ops.Intersection([stomach_raw, in_dome, abd])
    organ("stomach", "胃", "Ventriculus", sysd, stomach, "#e9a58e", voxel=0.8,
          note="左寄りにあるJ字形の単胃。噴門→胃底→胃体→幽門洞→幽門。空腹時は肋骨弓の内側に収まる。")
    gall = ell((80, -14, 206), (14.0, 8.5, 8.5), axis=V(1, -0.2, -0.4))
    liver = ops.Difference([
        ops.Intersection([in_dome, abd, ell((98, -6, 222), (64.0, 82.0, 66.0))]),
        ops.Offset(stomach_raw, 2.0), ops.Offset(gall, 0.8),
    ] + [Box((1.4, 40.0, 30.0), xform=(np.eye(3), V(80, y, 190)))
         for y in (30, 8, -22, -44)])
    organ("liver", "肝臓", "Hepar", sysd, liver, "#7a2a22", voxel=1.0,
          note="横隔膜のすぐ後ろ。犬の肝臓は6葉（左外側葉・左内側葉・方形葉・右内側葉・右外側葉・尾状葉）で、深い切れ込みで分かれる。")
    organ("gallbladder", "胆嚢", "Vesica fellea", sysd, gall, "#5e8f3a", voxel=0.5,
          note="方形葉と右内側葉のあいだにある。胆汁は総胆管から十二指腸へ出る。")
    duo_pts = [(80, -24, 218), (78, -40, 238), (60, -48, 254), (20, -48, 258), (-30, -46, 256),
               (-55, -32, 250), (-45, -10, 256), (-20, 6, 262), (-5, 14, 258)]
    organ("duodenum", "十二指腸", "Duodenum", sysd, tube(duo_pts, [7.0] * len(duo_pts)),
          "#f0b39a", voxel=0.6, note="幽門から右側を後ろへ（下行部）、後曲で折り返して左へ戻る（上行部）。")
    # 空腸: 腹側の空間を蛇行する
    jej = [(-5, 14, 258)]
    rows = [(238, 34), (226, -34), (214, 30), (202, -30), (212, 20)]
    xs = [10, -10, -30, -50, -70, -88]
    for j, (z, ysign) in enumerate(rows):
        ys = [ysign, -ysign]
        for i, x in enumerate(xs if j % 2 == 0 else xs[::-1]):
            jej.append((x, ys[i % 2] * (0.9 if i % 3 else 1.0), z - (i % 2) * 6))
    organ("jejunum", "空腸", "Jejunum", sysd,
          ops.Intersection([tube(jej, [6.5] * len(jej), n=5), abd]), "#f4bca6", voxel=0.6,
          note="小腸のいちばん長い部分。腸間膜につり下げられ、腹腔の腹側でとぐろを巻く。")
    ile = [jej[-1], (-60, -10, 222), (-35, -22, 236), (-16, -26, 246)]
    organ("ileum", "回腸", "Ileum", sysd, tube(ile, [6.0] * len(ile)), "#eaa88e", voxel=0.6,
          note="小腸の最後の部分。回結腸口で結腸につながる（盲腸の横）。")
    cec = [(-16, -26, 246), (-26, -38, 242), (-36, -36, 232), (-30, -26, 228), (-24, -30, 236)]
    organ("cecum", "盲腸", "Caecum", sysd, tube(cec, [7.5, 7.0, 6.5, 5.5, 4.5]), "#c9a27a",
          voxel=0.6, note="犬の盲腸は短く、コイル状（らせん状）に巻いている。右側、第2〜4腰椎の腹側。")
    col = [(-16, -26, 248), (22, -26, 258), (34, -10, 262), (34, 12, 262), (22, 26, 262),
           (-30, 26, 262), (-80, 16, 262), (-108, 4, 266)]
    organ("colon", "結腸", "Colon", sysd, tube(col, [9.0] * len(col)), "#c9a27a", voxel=0.7,
          note="上行結腸（右）→ 横行結腸（前）→ 下行結腸（左）の «？» 字形。犬の結腸は単純な形。")
    rec = [(-108, 4, 266), (-140, 0, 270), (-175, 0, 274), (-200, 0, 278)]
    organ("rectum", "直腸", "Rectum", sysd, tube(rec, [9.0, 9.0, 8.0, 6.0]), "#b98f68",
          voxel=0.6, note="骨盤腔を通って肛門へ。肛門の両側に肛門嚢がある。")
    panc = [(78, -40, 250), (60, -44, 262), (30, -46, 266), (0, -44, 266)]
    panc_l = [(78, -34, 250), (70, -14, 252), (60, 8, 254), (50, 28, 256)]
    organ("pancreas", "膵臓", "Pancreas", sysd,
          ops.Union([tube(panc, [6.5, 6.0, 5.0, 3.5]), tube(panc_l, [6.0, 5.5, 5.0, 3.5])]),
          "#f0d78c", voxel=0.6, note="V字形。右葉は十二指腸に沿い、左葉は胃の大弯の後ろを左へ伸びる。")

    # ── 脾臓（造血・リンパ系） ───────────────────────────────
    sp_o = V(62, 50, 266)
    sp_u = unit(V(-0.45, -0.25, -1.0))
    spleen = Plate([(0, -10), (30, -14), (70, -12), (100, -8), (108, 0), (100, 8), (70, 13),
                    (30, 14), (0, 9)],
                   lambda x, y: 9.0 - 3.0 * np.clip(np.asarray(x) / 108.0, 0, 1),
                   corner_r=0.0, xform=(np.column_stack([sp_u, unit(np.cross(V(0, 1, 0), sp_u)),
                                                         np.cross(sp_u, unit(np.cross(V(0, 1, 0), sp_u)))]),
                                        sp_o))
    organ("spleen", "脾臓", "Lien", "造血・リンパ系", ops.Intersection([ops.Offset(spleen, 1.5), abd]),
          "#8a2e4a", voxel=0.7, note="胃の大弯に沿って左側にある細長い舌状の臓器。血液の貯蔵・古い赤血球の処理。")

    # ── 泌尿器 ──────────────────────────────────────────────
    sysu = "泌尿器"

    def kidney(c, sg):
        c = V(*c)
        body = Ellipsoid((26.0, 15.0, 13.0), xform=(np.eye(3), c))
        hil = Sphere(c + V(0, -sg * 15.0, 0), 7.0)
        return ops.Difference([body, hil]), c + V(0, -sg * 10.0, 0)

    kr, hr = kidney((52, -28, 286), -1)
    kl, hl = kidney((14, 28, 290), 1)
    organ("kidney_r", "右腎", "Ren dexter", sysu, kr, "#8c3b2e", voxel=0.6,
          note="右腎は左腎より前（第13胸椎〜第1腰椎）。前端は肝臓の尾状葉の腎圧痕に入る。犬の腎臓は豆形の単腎。")
    organ("kidney_l", "左腎", "Ren sinister", sysu, kl, "#8c3b2e", voxel=0.6,
          note="左腎は第1〜3腰椎の高さ。右腎より半個ぶん後ろ。")
    bl_c = V(-92, 0, 230)
    bladder = ops.SmoothUnion([Ellipsoid((30.0, 22.0, 20.0), xform=(np.eye(3), bl_c)),
                               RoundCone(bl_c + V(-22, 0, 6), bl_c + V(-42, 0, 14), 7.0, 3.5)], k=6.0)
    organ("bladder", "膀胱", "Vesica urinaria", sysu, bladder, "#e9d27a", voxel=0.6,
          note="骨盤腔の入口〜後腹部の腹側。尿がたまると前へふくらむ。")
    organ("ureter_r", "右尿管", "Ureter dexter", sysu,
          tube([hr, (20, -20, 290), (-40, -14, 286), (-80, -10, 250), (-102, -6, 238)], [1.8] * 5),
          "#d9b46a", voxel=0.4, note="腎盤から膀胱の背側（膀胱三角）へ尿を運ぶ管。")
    organ("ureter_l", "左尿管", "Ureter sinister", sysu,
          tube([hl, (-20, 18, 292), (-60, 14, 284), (-86, 10, 250), (-102, 6, 238)], [1.8] * 5),
          "#d9b46a", voxel=0.4, note="腎盤から膀胱の背側（膀胱三角）へ尿を運ぶ管。")

    # ── 内分泌 ──────────────────────────────────────────────
    syse = "内分泌"
    v3 = B.VERT["C3"]
    th_c = v3["c"] - v3["up"] * 26.0 + V(0, 8.5, 0)
    organ("thyroid", "甲状腺", "Glandula thyroidea", syse,
          ell(tuple(th_c), (11.0, 3.0, 4.0), axis=v3["d"]), "#a83a3a", voxel=0.4, paired=True,
          note="気管の前部（第1〜8気管輪）の両側にある左右1対の腺。犬では左右の葉は普通つながらない。")
    organ("adrenal_r", "右副腎", "Glandula adrenalis dextra", syse,
          ell((80, -16, 290), (9.0, 3.5, 4.0)), "#e6b84a", voxel=0.35,
          note="右腎の前内側、後大静脈のすぐ横。")
    organ("adrenal_l", "左副腎", "Glandula adrenalis sinistra", syse,
          ell((42, 16, 292), (9.0, 3.5, 4.0)), "#e6b84a", voxel=0.35,
          note="左腎の前内側、大動脈のすぐ横。")

    # ── 生殖器（雄） ────────────────────────────────────────
    sysg = "生殖器（雄）"
    organ("prostate", "前立腺", "Prostata", sysg, Sphere(bl_c + V(-46, 0, 14), 9.0), "#d9a6c8",
          voxel=0.5, note="膀胱頸を取り巻く。犬の副生殖腺は前立腺だけ（精嚢腺・尿道球腺を欠く）。")
    hip = B.HIND["hip"]
    organ("testis", "精巣", "Testis", sysg,
          ell((hip[0] - 48, 11, 212), (15.0, 9.0, 10.0), axis=V(1, 0, -0.6)), "#e7c2b0",
          voxel=0.5, paired=True, note="陰嚢の中。両大腿のあいだ、肛門の腹側寄りにある。")

    # ── 横隔膜 ──────────────────────────────────────────────
    dia = ops.Intersection([
        ops.Difference([dome(1.6), dome(-1.6)]),
        half(DOME_C + V(8, 0, 0), V(-1, 0, 0)),
        ops.Offset(thorax_cavity(), 9.0),
    ])
    organ("diaphragm", "横隔膜", "Diaphragma", "その他", dia, "#b85450", voxel=0.8,
          note="胸腔と腹腔を分けるドーム状の筋。食道裂孔・大動脈裂孔・大静脈孔が通る。呼吸の主役。")
