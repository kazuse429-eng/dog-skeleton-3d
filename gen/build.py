# -*- coding: utf-8 -*-
"""骨ごとにメッシュを作り、ビューア用のデータ（../bones_data.js）に書き出す。

実行（Blender 4.x 同梱の Python。numpy と openvdb が入っている）:
    blender --background --python gen/build.py
    blender --background --python gen/build.py -- --only skull,humerus   # 一部だけ試す

出力の形式（bones_data.js）:
    window.DOG_BONES = { scale, bones: [ {id, ja, la, path, paired, note,
                                          o:[x,y,z], q:量子化の刻み, v:Int16の座標(base64),
                                          i:Uint16の三角形(base64)} ] }
座標は mm、+X 頭側 / +Y 左 / +Z 上。ビューアで Y-up に回す。
"""

import base64
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bones as BN                          # noqa: E402
from sdf import grid as G                   # noqa: E402
from sdf import mesh as MSH                 # noqa: E402

# 三角形1枚あたりの目安面積（mm²）。小さいほど細かい。
TRI_AREA = 2.2
TRI_MIN, TRI_MAX = 160, 9000


def decimate(P, Q, target):
    """Blender の Decimate で三角形数を target 付近まで減らす。"""
    import bpy
    import bmesh

    obj = MSH.to_blender("tmp", P, Q)
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    n0 = len(me.polygons)
    if n0 > target:
        mod = obj.modifiers.new("dec", "DECIMATE")
        mod.ratio = max(0.01, target / float(n0))
        mod.use_collapse_triangulate = True
        dg = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        me2 = bpy.data.meshes.new_from_object(ev)
    else:
        me2 = me.copy()
    me2.calc_loop_triangles()
    V = np.array([v.co[:] for v in me2.vertices], dtype=np.float64)
    T = np.array([t.vertices[:] for t in me2.loop_triangles], dtype=np.int64)
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(me2)
    return V, T


def area(V, T):
    a = V[T[:, 1]] - V[T[:, 0]]
    b = V[T[:, 2]] - V[T[:, 0]]
    return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def pack(V, T):
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    q = float(max(hi - lo)) / 65000.0
    q = max(q, 1e-4)
    Vi = np.round((V - lo) / q - 32500).astype(np.int16)
    if len(V) > 65535:
        raise ValueError("頂点が多すぎる")
    return (lo.tolist(), q, base64.b64encode(Vi.tobytes()).decode("ascii"),
            base64.b64encode(T.astype(np.uint16).tobytes()).decode("ascii"))


PARTS = os.path.join(HERE, "_parts")


def quad_area(P, Q):
    T = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]]).astype(np.int64)
    return area(P, T), len(T)


def make_one(b):
    t = time.time()
    f = b["field"]
    g = G.grid_for(f, b["voxel"])
    P, Q, _ = MSH.field_to_quads(f, g)
    if len(Q) == 0:
        print("!! 空のメッシュ", b["id"])
        return
    A, n0 = quad_area(P, Q)
    target = int(min(TRI_MAX, max(TRI_MIN, A / TRI_AREA)))
    V, T = decimate(P, Q, target)
    o, q, vs, ts = pack(V, T)
    rec = dict(id=b["id"], ja=b["ja"], la=b["la"], path=b["path"],
               paired=b["paired"], note=b["note"], o=o, q=q, v=vs, i=ts)
    with open(os.path.join(PARTS, b["id"] + ".json"), "w", encoding="utf-8") as fp:
        json.dump(rec, fp, ensure_ascii=False)
    print("%-12s %-22s 面積%7.0f mm2  三角 %6d → %5d  %5.1f秒" % (
        b["id"], b["ja"], A, n0, len(T), time.time() - t))


def merge(bones):
    """_parts の骨を定義順に並べて bones_data.js にする。"""
    out, total, missing = [], 0, []
    for b in bones:
        fn = os.path.join(PARTS, b["id"] + ".json")
        if not os.path.exists(fn):
            missing.append(b["id"])
            continue
        with open(fn, encoding="utf-8") as fp:
            rec = json.load(fp)
        # 名前・説明は定義から取り直す（メッシュを作り直さずに文言だけ直せる）
        rec.update(ja=b["ja"], la=b["la"], path=b["path"], paired=b["paired"], note=b["note"])
        total += len(base64.b64decode(rec["i"])) // 6
        out.append(rec)
    out_path = os.path.join(HERE, "..", "bones_data.js")
    js = "window.DOG_BONES = " + json.dumps(dict(units="mm", bones=out),
                                             ensure_ascii=False, separators=(",", ":")) + ";\n"
    with open(out_path + ".tmp", "w", encoding="utf-8") as fp:
        fp.write(js)
    os.replace(out_path + ".tmp", out_path)
    print("骨 %d 個 / 三角 %d / %.1f MB / 未作成 %s" % (len(out), total, len(js) / 1e6,
                                                   ",".join(missing) or "なし"))


def main():
    """引数:
        --only a,b     この骨だけ作る
        --shard k/n    定義順で i % n == k の骨だけ作る（並列実行用）
        --merge        作らずに _parts をまとめるだけ
    何も付けなければ全部作ってまとめる。
    """
    sys.stdout.reconfigure(line_buffering=True)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    os.makedirs(PARTS, exist_ok=True)
    bones = BN.build_all()
    if "--merge" in argv:
        merge(bones)
        return
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
    shard = None
    if "--shard" in argv:
        k, n = argv[argv.index("--shard") + 1].split("/")
        shard = (int(k), int(n))
    t0 = time.time()
    for i, b in enumerate(bones):
        if only and b["id"] not in only:
            continue
        if shard and i % shard[1] != shard[0]:
            continue
        make_one(b)
    print("完了 %.0f 秒" % (time.time() - t0))
    if not shard:
        merge(bones)


main()
