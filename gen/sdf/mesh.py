# -*- coding: utf-8 -*-
"""距離場からメッシュを取り出す。

OpenVDB（Blender 同梱）の `copyFromArray` + `convertToQuads` を使う。
マーチングキューブを自前実装しない理由:

- `convertToQuads` は閉じた四角メッシュを返す。多様体性が構造的に出る。
- 実測で、自前の球の距離場から作ったメッシュが
  `createLevelSetSphere`（OpenVDB 自身の球）と**体積が完全一致**した。
  サンプリングの質は OpenVDB 自身と同等ということ。
- スラブに切って `copyFromArray(ijk=...)` で流し込んでも、
  一括で流し込んだ場合と**1頂点も違わない**（実測で確認）。

実測しておいた落とし穴
----------------------
- `convertToQuads` が返す面は **内向き**（符号つき体積が負）。必ず反転する。
- グリッドの `transform` は `voxelSize` だけにして、原点のずれは
  返ってきた点に足す。`createLinearTransform` に行列を渡す形は使わない。
- 背景値をナローバンド幅にし、配列をその範囲にクランプしてから渡すと、
  遠方が非活性になって疎なグリッドになる（球で 50733/132651 voxel）。
"""

import math

import numpy as np

from .prim import FLOAT
from .grid import sample_slab, slabs


def field_to_quads(field, grid, band=None, slab_width=64, isovalue=0.0,
                   progress=None):
    """距離場 → (点 (N,3) float64 世界座標, 四角面 (M,4) uint32)。

    面の向きは**外向き**に直して返す。
    """
    import openvdb as vdb

    band = grid.voxel * 3.0 if band is None else float(band)
    g = vdb.FloatGrid(FLOAT(band))
    g.transform = vdb.createLinearTransform(voxelSize=grid.voxel)
    g.gridClass = vdb.GridClass.LEVEL_SET

    for i0, i1 in slabs(grid, slab_width):
        d = sample_slab(field, grid, i0, i1, band)
        np.clip(d, -band, band, out=d)
        g.copyFromArray(np.ascontiguousarray(d), ijk=(i0, 0, 0), tolerance=0.0)
        del d
        if progress:
            progress(i1, grid.shape[0])

    pts, quads = g.convertToQuads(isovalue)
    P = pts.astype(np.float64) + grid.origin
    # 内向きで返るので反転する
    Q = quads[:, ::-1].astype(np.uint32, copy=True)
    return P, Q, g


def quads_to_tris(quads):
    q = np.asarray(quads)
    return np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], axis=0)


def mesh_stats(points, quads):
    """体積・表面積・境界箱。符号つき体積が正なら面が外向き。"""
    P = np.asarray(points, dtype=np.float64)
    tris = quads_to_tris(quads)
    A, B, C = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    cr = np.cross(B - A, C - A)
    signed = float(np.einsum("ij,ij->i", A, cr).sum() / 6.0)
    return {
        "points": len(P),
        "quads": len(quads),
        "volume": abs(signed),
        "signed_volume": signed,
        "area": float(0.5 * np.linalg.norm(cr, axis=1).sum()),
        "bbox_lo": P.min(axis=0),
        "bbox_hi": P.max(axis=0),
        "size": P.max(axis=0) - P.min(axis=0),
    }


def to_blender(name, points, quads, scale=1.0):
    """Blender のメッシュオブジェクトにする。scale は最後に一度だけ掛ける。"""
    import bpy

    P = np.asarray(points, dtype=np.float64) * float(scale)
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in P], [], [tuple(int(i) for i in q) for q in quads])
    me.update()
    me.validate(verbose=False)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def save_stl(obj, path):
    import bpy

    for o in bpy.context.scene.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(filepath=str(path), export_selected_objects=True,
                          global_scale=1.0, apply_modifiers=True)
    return path
