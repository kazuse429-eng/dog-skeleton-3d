# -*- coding: utf-8 -*-
"""実物スキャンの骨に合わせて、全体を組み直す（bones_data.js の最終段）。

    python gen/realfit.py

流れ
----
1. 実物スキャン（assets/scan/*.npz、Massey University 3D Dog Bone Project、CC BY-SA 4.0）
   から関節などの目印点を自動で取る。
2. 模型（gen/_parts の骨・臓器）の同じ目印点との対応から、薄板スプライン（TPS）で
   «模型の空間 → 実物の空間» のなめらかな変形を作り、模型の全頂点を動かす。
   これで、実物が無い骨や臓器も実物の骨格の中の正しい場所に来る。
3. 実物がある骨はスキャンに差し替える。
   - 腰椎 L2〜L7 は L1 のスキャン、頸椎 C3〜C6 は C7 のスキャンを、各位置・大きさに合わせて流用。
   - 肋骨 13 対は、実物の第1・4・9・13肋骨のうち近いものを、各肋骨の両端に合わせて変形。
   - 寛骨は1個のスキャンを腸骨・坐骨・恥骨の範囲で切り分ける（境界は目安）。
4. 各骨に «出どころ»（scan / derived / model）と出典を付ける。

座標は mm、+X 頭側 / +Y 左 / +Z 上、地面 Z=0。
"""

import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)

import bones as BN  # noqa: E402

PARTS = os.path.join(HERE, "_parts")
SCAN = os.path.join(ROOT, "assets", "scan")

CREDIT_MASSEY = {
    "src": "scan",
    "credit": "Massey University 3D Dog Bone Project (nzfauna), CC BY-SA 4.0",
    "url": "https://sketchfab.com/3d-models/3d-dog-bone-project-partial-dog-skeleton-65a6c5db1c4c477aba475af5d86d59a5",
}


# ─── 入出力 ────────────────────────────────────────────────────────────

def decode(rec):
    vi = np.frombuffer(base64.b64decode(rec["v"]), dtype=np.int16).reshape(-1, 3).astype(np.float64)
    V = np.asarray(rec["o"]) + (vi + 32500.0) * rec["q"]
    raw = base64.b64decode(rec["i"])
    T = np.frombuffer(raw, dtype=np.uint16 if len(V) <= 65535 else np.uint32).reshape(-1, 3)
    return V, T.astype(np.int64)


def encode(rec, V, T):
    lo = V.min(0)
    q = max(float(max(V.max(0) - lo)) / 65000.0, 1e-4)
    Vi = np.round((V - lo) / q - 32500).astype(np.int16)
    if len(V) > 65535:
        raise ValueError("頂点が多すぎる: %s" % rec["id"])
    rec.update(o=lo.tolist(), q=q, v=base64.b64encode(Vi.tobytes()).decode("ascii"),
               i=base64.b64encode(T.astype(np.uint16).tobytes()).decode("ascii"))
    return rec


def scan(name):
    d = np.load(os.path.join(SCAN, name + ".npz"))
    return d["V"].astype(np.float64), d["T"].astype(np.int64)


# ─── 目印点 ───────────────────────────────────────────────────────────

def end_centroid(V, axis, side, frac=0.06):
    """axis 方向の端（side=+1 で最大側）の frac ぶんの点の重心。"""
    s = V @ np.asarray(axis, dtype=float) * side
    k = max(10, int(len(V) * frac))
    idx = np.argsort(-s)[:k]
    return V[idx].mean(0)


def real_landmarks():
    L = {}
    for k in ("C1", "C7", "L1"):
        V, _ = scan(k)
        L[k] = V.mean(0) * [1, 0, 1]           # 正中へ
    hu, _ = scan("humerus")
    L["glenoid"] = end_centroid(hu, (0, 0, 1), +1)
    L["elbow"] = end_centroid(hu, (0, 0, 1), -1)
    ra, _ = scan("radius")
    L["carpus"] = end_centroid(ra, (0, 0, 1), -1)
    sc, _ = scan("scapula")
    L["scap_top"] = end_centroid(sc, (0, 0, 1), +1)
    fe, _ = scan("femur")
    L["hip"] = end_centroid(fe, (0, 0, 1), +1)
    L["stifle"] = end_centroid(fe, (0, 0, 1), -1)
    ti, _ = scan("tibia")
    L["hock"] = end_centroid(ti, (0, 0, 1), -1)
    hb, _ = scan("ilium_all")
    L["tuber_coxae"] = end_centroid(hb, (1, 0, 0), +1)
    L["tuber_isch"] = end_centroid(hb, (1, 0, 0), -1)
    sk, _ = scan("skull")
    L["nose"] = end_centroid(sk, (1, 0, 0), +1, 0.02) * [1, 0, 1]
    L["occiput"] = end_centroid(sk, (1, 0, 0), -1, 0.03) * [1, 0, 1]
    return L


def model_landmarks():
    F, Hd = BN.FORE, BN.HIND
    L = {k: BN.VERT[k]["c"].copy() for k in ("C1", "C7", "L1")}
    L["glenoid"] = F["glenoid"]
    L["elbow"] = F["elbow"]
    L["carpus"] = F["carpus_top"]
    L["scap_top"] = F["scap_top"]
    L["hip"] = Hd["hip"]
    L["stifle"] = Hd["stifle"]
    L["hock"] = Hd["hock"]
    L["tuber_coxae"] = Hd["tuber_coxae"]
    L["tuber_isch"] = Hd["tuber_isch"]
    L["nose"] = BN.H(176, 0, 0) * [1, 0, 1]
    L["occiput"] = BN.H(-4, 2, 0) * [1, 0, 1]
    return L


# ─── 薄板スプライン（3D、核は r） ───────────────────────────────────────

class TPS(object):
    def __init__(self, src, dst, lam=0.0):
        self.src = np.asarray(src, dtype=float)
        n = len(src)
        K = np.linalg.norm(self.src[:, None] - self.src[None], axis=2) + lam * np.eye(n)
        P = np.hstack([np.ones((n, 1)), self.src])
        A = np.zeros((n + 4, n + 4))
        A[:n, :n] = K
        A[:n, n:] = P
        A[n:, :n] = P.T
        b = np.zeros((n + 4, 3))
        b[:n] = dst
        self.w = np.linalg.solve(A, b)

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X)
        n = len(self.src)
        for i in range(0, len(X), 20000):
            x = X[i:i + 20000]
            U = np.linalg.norm(x[:, None] - self.src[None], axis=2)
            out[i:i + 20000] = U @ self.w[:n] + self.w[n] + x @ self.w[n + 1:]
        return out


def build_warp():
    R, M = real_landmarks(), model_landmarks()
    keys = sorted(R)
    src, dst = [], []
    for k in keys:
        src.append(M[k]); dst.append(R[k])
        if abs(M[k][1]) > 5:                      # 左右対称になるよう右側にも同じ点
            src.append(M[k] * [1, -1, 1]); dst.append(R[k] * [1, -1, 1])
    # 足先（実物に無い）: 手首・かかとの真下の地面を対応させ、下で暴れないようにする
    for a, b in (("carpus", 24.0), ("hock", 30.0)):
        for sg in (1, -1):
            src.append(M[a] * [1, sg, 0] + [b, 0, 0])
            dst.append(R[a] * [1, sg, 0] + [b * 1.2, 0, 0])
    tps = TPS(np.array(src), np.array(dst), lam=0.0)
    err = np.abs(tps(np.array(src)) - np.array(dst)).max()
    print("TPS 目印 %d 点、当てはめ誤差 %.3f mm" % (len(src), err))
    return tps, R, M


# ─── 実物の形を各位置へ ─────────────────────────────────────────────────

def frame_of(V):
    """主成分の枠（重心、軸3本、各軸の広がり）。軸の符号は +X/+Z を正に揃える。"""
    c = V.mean(0)
    U, S, Wt = np.linalg.svd(V - c, full_matrices=False)
    return c, Wt


def place_vertebra(Vs, slot_c, slot_d, slot_up, scale):
    """スキャンの椎骨を、頭尾方向 slot_d・背側 slot_up の位置へ。スキャンは +X 頭・+Z 背のまま。"""
    c = Vs.mean(0)
    d = -np.asarray(slot_d)              # bones の d は尾側向き → 頭側向きへ
    up = np.asarray(slot_up)
    left = np.cross(up, d)
    Rm = np.column_stack([d, left, up])  # スキャンの (X, Y, Z) → (頭, 左, 背)
    return (Vs - c) * scale @ Rm.T + slot_c


def rib_frame(V):
    """肋骨の «頭（椎骨側）»・«腹側端»・«いちばん外の点»。"""
    head = end_centroid(V, (0, -0.4, 1), +1, 0.03)
    tip = end_centroid(V, (0, 0, 1), -1, 0.03)
    lat = end_centroid(V, (0, 1, 0), +1, 0.03)
    return head, tip, lat


def fit_rib(Vs, tgt_head, tgt_tip, tgt_lat):
    """3点の枠を合わせる（各軸を別々に伸縮）。"""
    h, t, l = rib_frame(Vs)

    def basis(a, b, c):
        e1 = b - a
        L1 = np.linalg.norm(e1)
        e1 = e1 / L1
        w = c - a
        e2 = w - e1 * (w @ e1)
        L2 = np.linalg.norm(e2)
        e2 = e2 / L2
        return e1, e2, np.cross(e1, e2), L1, L2

    a1, a2, a3, la1, la2 = basis(h, t, l)
    b1, b2, b3, lb1, lb2 = basis(tgt_head, tgt_tip, tgt_lat)
    s1 = lb1 / la1
    # 横の張り出しの倍率は縦の倍率の 0.6〜1.8 倍に収める。
    # ほぼまっすぐな第1肋骨を曲がった肋骨に合わせたとき、ここが何倍にもなって板状に潰れた。
    s2 = float(np.clip(lb2 / max(la2, 1e-6), s1 * 0.6, s1 * 1.8))
    s3 = (s1 + s2) * 0.5
    X = Vs - h
    loc = np.stack([X @ a1 * s1, X @ a2 * s2, X @ a3 * s3], axis=1)
    return tgt_head + loc[:, :1] * b1 + loc[:, 1:2] * b2 + loc[:, 2:] * b3


# ─── 臓器の当てはめ（向き・大きさ） ─────────────────────────────────────

CREDIT_ORGAN = {
    "liver": dict(src="scan", credit="\"Dog, Liver\" by 3DVet_Printing (CT), CC BY 4.0",
                  url="https://sketchfab.com/3d-models/dog-liver-b0836f9e9ec645a0971f7fb9629d7e44"),
    "heart": dict(src="scan", credit="\"Normal Canine Heart\" by atugman, CC BY 4.0",
                  url="https://sketchfab.com/3d-models/normal-canine-heart-acb173cf049448d4ae927ddea2a89753"),
}


def _proper_rotations():
    """軸の入れ替え・符号の 48 通り。鏡像（臓器の左右が逆）は呼び出し側で R の行列式で除く。"""
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, perm[i]] = sg[i]
            out.append(M)          # 鏡像は後で R の行列式で除く（主軸の枠そのものの向きで決まるため）
    return out


def fit_organ(Vs, Vt, iters=25):
    """スキャンの臓器 Vs を、模型の同じ臓器 Vt に重ねる。

    主成分の枠で大まかに合わせ、24 通りの向きを全部試して «面同士の平均距離» が
    最小のものを選び（鏡像は除く）、そこから ICP（回転＋一様な拡大縮小）で詰める。
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    sv = Vs[rng.choice(len(Vs), min(3000, len(Vs)), replace=False)]
    tv = Vt[rng.choice(len(Vt), min(3000, len(Vt)), replace=False)]
    cs, ct = Vs.mean(0), Vt.mean(0)
    _, Ss, Ws = np.linalg.svd(sv - cs, full_matrices=False)
    _, St, Wt = np.linalg.svd(tv - ct, full_matrices=False)
    scale = float(np.sqrt((St ** 2).sum() / (Ss ** 2).sum()))
    tree_t = cKDTree(tv)
    best = None
    for M in _proper_rotations():
        R = Wt.T @ M @ Ws                   # スキャンの主軸 → 模型の主軸（入れ替え・符号違い）
        if np.linalg.det(R) < 0:
            continue
        X = (sv - cs) @ R.T * scale + ct
        d = tree_t.query(X)[0].mean() + cKDTree(X).query(tv)[0].mean()
        if best is None or d < best[0]:
            best = (d, R)
    R, s, t = best[1], scale, ct - cs @ best[1].T * scale
    # ICP（相似変換）
    for _ in range(iters):
        X = sv @ R.T * s + t
        _, j = tree_t.query(X)
        Y = tv[j]
        mx, my = sv.mean(0), Y.mean(0)
        A, B = sv - mx, Y - my
        U, S, Vt2 = np.linalg.svd(A.T @ B)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt2.T @ U.T))])
        R = Vt2.T @ D @ U.T
        s = float((S * np.diag(D)).sum() / (A ** 2).sum())
        t = my - mx @ R.T * s
    err = tree_t.query(sv @ R.T * s + t)[0].mean()
    return Vs @ R.T * s + t, err


# ─── 脳（コーネル大学 MRI アトラス、CC0） ─────────────────────────────────

CREDIT_BRAIN = dict(src="scan", credit="Johnson & Barry (2019) Cortical Atlas of the Canine Brain, Cornell University (MRI), CC0",
                    url="https://doi.org/10.7298/4t8z-aw34")

BRAIN = [  # id, 和名, ラテン名, 色, 左右, 説明
    ("brain_frontal", "前頭葉", "Lobus frontalis", "#e9a3a8", True,
     "大脳の前部。十字溝より前で、行動の計画などに関わる。"),
    ("brain_sensorimotor", "感覚運動野（十字溝周囲）", "Cortex sensorimotorius", "#e7b77c", True,
     "十字溝のまわりの皮質。一次運動野と体性感覚野がある（犬の運動野は前方の十字溝周囲）。"),
    ("brain_parietal", "頭頂葉", "Lobus parietalis", "#d9c36e", True,
     "大脳の背側中央。感覚情報の統合。"),
    ("brain_temporal", "側頭葉（外シルビウス溝周囲）", "Lobus temporalis", "#9fc98a", True,
     "大脳の外側、外シルビウス溝のまわり。聴覚野がある。"),
    ("brain_occipital", "後頭葉", "Lobus occipitalis", "#8fb5df", True,
     "大脳の後部。視覚野（外側回・後外側回の領域）がある。"),
    ("brain_cingulate", "帯状回", "Gyrus cinguli", "#c39bd6", True,
     "大脳半球の内側面、脳梁の上。辺縁系の一部（情動・記憶）。"),
    ("brain_olfactory", "嗅球", "Bulbus olfactorius", "#f0c48a", True,
     "大脳の前端。犬は嗅覚が発達していて嗅球が大きい。"),
    ("brain_white", "大脳髄質（白質）", "Substantia alba", "#ece4d6", True,
     "皮質の内側の白質。表面の脳葉のすき間から見える部分（深部の核は含めて表示）。"),
    ("brain_cerebellum", "小脳", "Cerebellum", "#d4949c", False,
     "大脳の後ろ、脳幹の背側。中央の虫部と左右の半球。運動の協調・平衡。"),
    ("brain_stem", "脳幹", "Truncus encephali", "#c98b8b", False,
     "中脳・橋・延髄（この表示は間脳の一部を含む目安）。大後頭孔を通って脊髄へ続く。"),
]


def fit_brain(target):
    """脳全体（左右）を、模型の脳（曲げ済み）に «前後の傾き・拡大・平行移動» だけで合わせる。

    左右を崩さない（右半球はビューアで鏡像にする）ため、回転は左右軸まわりだけ。
    """
    from scipy.spatial import cKDTree
    parts = {k: scan(k) for k, *_ in BRAIN}
    allV = []
    for k, ja, la, col, paired, note in BRAIN:
        V, _ = parts[k]
        allV.append(V)
        if paired:
            allV.append(V * [1, -1, 1])
    A = np.concatenate(allV)
    rng = np.random.default_rng(1)
    A = A[rng.choice(len(A), 4000, replace=False)]
    tv = target[rng.choice(len(target), min(4000, len(target)), replace=False)]
    tree = cKDTree(tv)
    ct = tv.mean(0) * [1, 0, 1]
    s0 = (np.ptp(tv[:, 0]) / np.ptp(A[:, 0]))
    best = None
    for ang in np.radians(np.arange(-45, 46, 3)):
        for s in s0 * np.array([0.85, 0.92, 1.0, 1.08, 1.16]):
            c, sn = np.cos(ang), np.sin(ang)
            R = np.array([[c, 0, -sn], [0, 1, 0], [sn, 0, c]])
            X = (A - A.mean(0) * [1, 0, 1]) @ R.T * s + ct
            d = tree.query(X)[0].mean() + cKDTree(X).query(tv)[0].mean()
            if best is None or d < best[0]:
                best = (d, R, s)
    d, R, s = best
    off = ct - (A.mean(0) * [1, 0, 1]) @ R.T * s
    return parts, (lambda V: V @ R.T * s + off), d


# ─── 本体 ─────────────────────────────────────────────────────────────

def main():
    bones = BN.build_all()
    recs = {}
    for b in bones:
        fn = os.path.join(PARTS, b["id"] + ".json")
        with open(fn, encoding="utf-8") as fp:
            rec = json.load(fp)
        rec.update(ja=b["ja"], la=b["la"], path=b["path"], paired=b["paired"], note=b["note"],
                   layer=b["layer"], color=b["color"], alpha=b["alpha"], src="model")
        recs[b["id"]] = rec

    tps, R, M = build_warp()

    # 1) 模型を全部、実物の空間へ曲げる
    for rec in recs.values():
        V, T = decode(rec)
        encode(rec, tps(V), T)

    # 2) 実物に差し替え
    for key in ("skull", "mandible", "scapula", "humerus", "radius", "ulna", "femur", "tibia", "fibula"):
        V, T = scan(key)
        encode(recs[key], V, T)
        recs[key].update(CREDIT_MASSEY)
    for key in ("C1", "C7", "L1"):
        V, T = scan(key)
        encode(recs[key], V, T)
        recs[key].update(CREDIT_MASSEY)

    # 寛骨: 1個のスキャンを3つに切り分ける（三角形の重心で振り分け）
    V, T = scan("ilium_all")
    fe, _ = scan("femur")
    acet = end_centroid(fe, (0, -0.5, 1), +1, 0.03)          # 大腿骨頭 ≒ 寛骨臼
    C = V[T].mean(1)
    il = C[:, 0] > acet[0] + 6.0
    pub = (~il) & (C[:, 0] > acet[0] - 38.0) & (C[:, 2] < acet[2] - 6.0) & (C[:, 1] < acet[1] - 6.0)
    isc = ~(il | pub)
    for key, mask in (("ilium", il), ("pubis", pub), ("ischium", isc)):
        TT = T[mask]
        used = np.unique(TT)
        remap = -np.ones(len(V), dtype=np.int64)
        remap[used] = np.arange(len(used))
        encode(recs[key], V[used], remap[TT])
        recs[key].update(CREDIT_MASSEY)
        recs[key]["note"] += "（実物スキャンを腸骨・坐骨・恥骨の範囲で切り分けて表示。境界は目安）"

    # 腰椎 L2〜L7 ← L1、頸椎 C3〜C6 ← C7（模型の位置・向き・長さに合わせる）
    for base, targets in (("L1", ["L%d" % i for i in range(2, 8)]), ("C7", ["C%d" % i for i in range(3, 7)])):
        Vs, Ts = scan(base)
        vb = BN.VERT[base]
        for key in targets:
            v = BN.VERT[key]
            c = tps(v["c"][None])[0] * [1, 0, 1]
            d = tps((v["c"] + v["d"] * 5.0)[None])[0] - tps((v["c"] - v["d"] * 5.0)[None])[0]
            d = d / np.linalg.norm(d)
            up = np.array([0.0, 0.0, 1.0])
            up = up - d * (up @ d)
            up = up / np.linalg.norm(up)
            scale = (v["L"] / vb["L"]) ** 0.8
            encode(recs[key], place_vertebra(Vs, c, d, up, scale), Ts)
            recs[key].update(src="derived", credit=CREDIT_MASSEY["credit"], url=CREDIT_MASSEY["url"],
                             derived_from={"L1": "第1腰椎", "C7": "第7頸椎"}[base])

    # 肋骨 13 本 ← 実物の第1・4・9・13肋骨のうち近いもの
    real_ribs = {n: scan("rib%d" % n) for n in (1, 4, 9, 13)}
    for n in range(1, 14):
        key = "rib%d" % n
        Vm, _ = decode(recs[key])                         # 曲げ済みの模型の肋骨
        h, t, l = rib_frame(Vm)
        # 第1肋骨は短くまっすぐで特殊なので、第2肋骨以降には使わない
        near = 1 if n == 1 else (4 if n <= 6 else (9 if n <= 11 else 13))
        Vs, Ts = real_ribs[near]
        encode(recs[key], fit_rib(Vs, h, t, l), Ts)
        if near == n:
            recs[key].update(CREDIT_MASSEY)
        else:
            recs[key].update(src="derived", credit=CREDIT_MASSEY["credit"], url=CREDIT_MASSEY["url"],
                             derived_from="第%d肋骨" % near)

    # 臓器: 肝臓・心臓を実物（CT）に
    for key, fname in (("liver", "organ_liver"), ("heart", "organ_heart_atugman")):
        Vt, _ = decode(recs[key])
        Vs, Ts = scan(fname)
        V2, err = fit_organ(Vs, Vt)
        encode(recs[key], V2, Ts)
        recs[key].update(CREDIT_ORGAN[key])
        print("臓器 %-6s 当てはめ後の平均のずれ %.1f mm" % (key, err))

    # 脳: 模型の大脳・小脳・脳幹を、アトラスの部位に置き換える
    tgt = np.concatenate([decode(recs[k])[0] for k in ("cerebrum", "cerebellum", "brainstem")]
                         + [decode(recs["cerebrum"])[0] * [1, -1, 1]])
    parts, place, d = fit_brain(tgt)
    print("脳 当てはめ後の平均のずれ %.1f mm" % (d / 2))
    brain_recs = []
    for k, ja, la, col, paired, note in BRAIN:
        V, T = parts[k]
        r = dict(id=k, ja=ja, la=la, path=["臓器", "神経系", "脳"], paired=paired, note=note,
                 layer="organ", color=col, alpha=1.0)
        encode(r, place(V), T)
        r.update(CREDIT_BRAIN)
        brain_recs.append(r)

    out = []
    for b in bones:
        if b["id"] == "cerebrum":
            out += brain_recs
        elif b["id"] in ("cerebellum", "brainstem"):
            continue
        else:
            out.append(recs[b["id"]])
    credits = [
        dict(what="実物スキャンの骨（頭蓋骨・下顎骨・環椎・第7頸椎・第1腰椎・肋骨4本・肩甲骨・上腕骨・橈骨・尺骨・寛骨・大腿骨・脛骨・腓骨）",
             who="Massey University 3D Dog Bone Project (nzfauna)", license="CC BY-SA 4.0",
             url=CREDIT_MASSEY["url"], mod="骨ごとに分割・間引き・座標変換。一部の骨は形を流用して変形"),
        dict(what="脳（大脳の各葉・帯状回・嗅球・白質・小脳・脳幹）", who="Johnson, P. J. & Barry, E. F. (2019) Cortical Atlas of the Canine Brain, Cornell University（30頭のMRI平均）",
             license="CC0 1.0", url=CREDIT_BRAIN["url"], mod="ラベルから表面を作成、部位の分け方（小脳・嗅球・脳幹）は位置による目安、頭蓋の中へ配置"),
        dict(what="肝臓（CT）", who="3DVet_Printing", license="CC BY 4.0", url=CREDIT_ORGAN["liver"]["url"],
             mod="間引き・体内の位置へ移動と拡大縮小"),
        dict(what="心臓（大血管の根元を含む）", who="atugman", license="CC BY 4.0", url=CREDIT_ORGAN["heart"]["url"],
             mod="間引き・体内の位置へ移動と拡大縮小"),
    ]
    js = "window.DOG_BONES = " + json.dumps(dict(units="mm", bones=out, credits=credits),
                                             ensure_ascii=False, separators=(",", ":")) + ";\n"
    path = os.path.join(ROOT, "bones_data.js")
    with open(path + ".tmp", "w", encoding="utf-8") as fp:
        fp.write(js)
    os.replace(path + ".tmp", path)
    n_scan = sum(1 for r in out if r.get("src") == "scan")
    n_der = sum(1 for r in out if r.get("src") == "derived")
    print("骨・臓器 %d 個（実物 %d、実物の形を流用 %d） / %.1f MB" % (len(out), n_scan, n_der, len(js) / 1e6))
    for k in sorted(R):
        print("  目印 %-12s 模型→実物 %s → %s" % (k, np.round(M[k], 0), np.round(R[k], 0)))


if __name__ == "__main__":
    main()
