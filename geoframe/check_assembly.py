"""組み上がった多面体で、近くにあるタイルどうしの干渉体積を測る

  uv run --with scipy --with numpy --with matplotlib --with manifold3d python check_assembly.py [名前の一部 ...]

辺を共有する2枚は geoframe.poly の干渉テストで確かめてあるが、頂点だけを共有する
3枚目以降は確かめていなかった。ここでは組み上げた状態の全ペアを調べる。
"""
import sys

import numpy as np
import manifold3d as m3d

from assemble import TILE, HERE, read_stl, assemble
from poly_list import S


def to_manifold(T):
    V, idx = np.unique(np.round(T.reshape(-1, 3), 5), axis=0, return_inverse=True)
    mesh = m3d.Mesh(vert_properties=V.astype(np.float32), tri_verts=idx.reshape(-1, 3).astype(np.uint32))
    return m3d.Manifold(mesh)


def main():
    tiles = {n: read_stl(HERE / f"{f}.stl") for n, f in TILE.items()}
    want = sys.argv[1:]
    for name, pts in S.items():
        if want and not any(w in name for w in want):
            continue
        parts = assemble(pts, tiles)
        ms = [to_manifold(T) for _, T in parts]
        bad = [i for i, m in enumerate(ms) if m.status() != m3d.Error.NoError]
        cs = [T.reshape(-1, 3).mean(0) for _, T in parts]
        rs = [np.linalg.norm(T.reshape(-1, 3) - c, axis=1).max() for (_, T), c in zip(parts, cs)]
        worst, total, pairs = 0.0, 0.0, 0
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                if np.linalg.norm(cs[i] - cs[j]) > rs[i] + rs[j]:
                    continue
                pairs += 1
                v = (ms[i] ^ ms[j]).volume()
                total += v
                worst = max(worst, v)
        flag = "OK" if worst < 0.5 else "NG"
        print(f"{flag} {name:28s} 枚数 {len(parts):3d} 近接ペア {pairs:4d}  干渉合計 {total:7.2f} mm3  最大 {worst:6.2f} mm3"
              + (f"  非多様体メッシュ {len(bad)}" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
