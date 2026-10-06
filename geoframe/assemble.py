"""組み上がった多面体を、タイルの STL を各面に配置して描画する

  uv run --with scipy --with numpy --with matplotlib python assemble.py [名前の一部 ...]

タイルの STL (triangle/square/pentagon/hexagon/octagon/decagon.stl) は geoframe.poly の座標系:
中心が原点、厚み z = 0..3、ヒンジ軸は z = 1.5、上辺 (+Y) が X 方向。
多面体の各面に、タイルの中心面 (z = 1.5) が面に一致し、法線が外向きになるよう置く。
"""
import math
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from poly_list import S, NG

HERE = Path(__file__).parent
EDGE = 35.0      # geoframe.poly の edge
MID = 1.5        # タイルの中心面 (thick / 2)
TILE = {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon", 8: "octagon", 10: "decagon"}
COLOR = {3: "#e4572e", 4: "#2e86ab", 5: "#57a773", 6: "#f3a712", 8: "#8e5ea2", 10: "#6c757d"}
KANJI = {3: "三", 4: "四", 5: "五", 6: "六", 8: "八", 10: "十"}

SLUG = {
    "正四面体": "tetrahedron", "立方体": "cube", "正八面体": "octahedron",
    "正十二面体": "dodecahedron", "正二十面体": "icosahedron",
    "切頂四面体": "truncated-tetrahedron", "立方八面体": "cuboctahedron",
    "切頂八面体": "truncated-octahedron", "斜方立方八面体": "rhombicuboctahedron",
    "二十・十二面体": "icosidodecahedron", "切頂二十面体(サッカーボール)": "truncated-icosahedron",
    "斜方二十・十二面体": "rhombicosidodecahedron", "変形立方体(スナブ立方体)": "snub-cube",
    "変形十二面体(スナブ十二面体)": "snub-dodecahedron",
    "三角柱": "triangular-prism", "五角柱": "pentagonal-prism", "六角柱": "hexagonal-prism",
    "正四角反柱": "square-antiprism", "正五角反柱": "pentagonal-antiprism", "正六角反柱": "hexagonal-antiprism",
    "切頂六面体": "truncated-cube", "切頂立方八面体": "truncated-cuboctahedron",
    "切頂十二面体": "truncated-dodecahedron", "切頂二十・十二面体": "truncated-icosidodecahedron",
    "八角柱": "octagonal-prism", "十角柱": "decagonal-prism",
    "正八角反柱": "octagonal-antiprism", "正十角反柱": "decagonal-antiprism",
}


def slug(name):
    if name in SLUG:
        return SLUG[name]
    return name.split()[0].lower()  # "J3 三角キューポラ" -> "j3"


def read_stl(path):
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].astype(float)  # (n, 3, 3)


def faces_of(pts):
    """凸包の面: (角数, 中心, 外向き法線, 辺の中点の1つ, 辺長)"""
    P = np.array(pts, float)
    hull = ConvexHull(P)
    planes, groups = [], []
    for eq, s in zip(hull.equations, hull.simplices):
        for i, q in enumerate(planes):
            if np.allclose(eq, q, atol=1e-6):
                groups[i] |= set(s)
                break
        else:
            planes.append(eq)
            groups.append(set(s))
    out = []
    for eq, g in zip(planes, groups):
        V = P[sorted(g)]
        c = V.mean(axis=0)
        d = np.linalg.norm(V[1:] - V[0], axis=1)
        e = d.min()
        nb = V[1:][np.argmin(d)]
        out.append((len(V), c, eq[:3], (V[0] + nb) / 2, e))
    return out


def assemble(pts, tiles):
    """[(角数, 三角形配列(world))]"""
    fs = faces_of(pts)
    scale = EDGE / fs[0][4]
    parts = []
    for n, c, nrm, m, _ in fs:
        z = nrm / np.linalg.norm(nrm)
        y = m - c
        y -= z * np.dot(y, z)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        R = np.column_stack([x, y, z])
        T = tiles[n].copy()
        T[..., 2] -= MID
        parts.append((n, T @ R.T + c * scale))
    return parts


def render(name, parts, path, elev=22, azim=-58):
    light = np.array([0.4, -0.6, 0.9])
    light /= np.linalg.norm(light)
    polys, cols = [], []
    for n, T in parts:
        nv = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        nv /= np.linalg.norm(nv, axis=1, keepdims=True) + 1e-12
        shade = 0.35 + 0.65 * np.abs(nv @ light)
        base = np.array(matplotlib.colors.to_rgb(COLOR[n]))
        polys.append(T)
        cols.append(np.clip(base[None, :] * shade[:, None], 0, 1))
    polys = np.concatenate(polys)
    cols = np.concatenate(cols)
    fig = plt.figure(figsize=(5, 5), dpi=120)
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none", linewidths=0))
    lo, hi = polys.reshape(-1, 3).min(0), polys.reshape(-1, 3).max(0)
    mid, r = (lo + hi) / 2, (hi - lo).max() / 2
    for set_lim, k in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        set_lim(mid[k] - r, mid[k] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    counts = {}
    for n, _ in parts:
        counts[n] = counts.get(n, 0) + 1
    label = " + ".join(f"{KANJI[n]}{counts[n]}" for n in sorted(counts))
    size = (hi - lo).max()
    fig.text(0.5, 0.04, f"{name}\n{label} = {len(parts)}枚   約{size / 10:.1f}cm",
             ha="center", fontsize=10, fontfamily=["Noto Sans CJK JP", "DejaVu Sans"])
    fig.subplots_adjust(0, 0.08, 1, 1)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    tiles = {n: read_stl(HERE / f"{f}.stl") for n, f in TILE.items()}
    outdir = HERE / "assembled"
    outdir.mkdir(exist_ok=True)
    want = sys.argv[1:]
    for name, pts in S.items():
        if name in NG:
            continue
        if want and not any(w in name for w in want):
            continue
        parts = assemble(pts, tiles)
        out = outdir / f"{slug(name)}.png"
        render(name, parts, out)
        print(f"{out.name}: {name} {len(parts)}枚")


if __name__ == "__main__":
    main()
