"""正3〜6角形だけでできる凸多面体の面数・二面角を座標から計算する"""
import itertools, math, json
import numpy as np
from scipy.spatial import ConvexHull

phi = (1 + 5 ** 0.5) / 2


def perms(v, even_only=False, signs=True):
    out = set()
    idx = [(0, 1, 2), (1, 2, 0), (2, 0, 1)] if even_only else list(itertools.permutations(range(3)))
    for p in idx:
        base = [v[i] for i in p]
        if signs:
            for s in itertools.product([1, -1], repeat=3):
                out.add(tuple(round(b * t, 9) for b, t in zip(base, s)))
        else:
            out.add(tuple(base))
    return [list(x) for x in out]


def ring(n, z, rot=0.0, edge=1.0):
    R = edge / (2 * math.sin(math.pi / n))
    return [[R * math.cos(rot + 2 * math.pi * i / n), R * math.sin(rot + 2 * math.pi * i / n), z] for i in range(n)]


def R(n):
    return 1 / (2 * math.sin(math.pi / n))


def apex_h(n):
    return math.sqrt(1 - R(n) ** 2)


def anti_h(n):
    d = 2 * R(n) * math.sin(math.pi / (2 * n))
    return math.sqrt(1 - d * d)


def cupola_h():
    # 三角キューポラ: 上の三角(半径 R3, 回転30°)と下の六角(半径1)
    t = np.array([R(3) * math.cos(math.pi / 6), R(3) * math.sin(math.pi / 6)])
    b = np.array([1.0, 0.0])
    b2 = np.array([math.cos(math.pi / 3), math.sin(math.pi / 3)])
    d = min(np.linalg.norm(t - b), np.linalg.norm(t - b2))
    return math.sqrt(1 - d * d)


def stack(layers):
    pts = []
    for l in layers:
        if l[0] == "apex":
            pts.append([0, 0, l[1]])
        else:
            n, z, rot = l
            pts += ring(n, z, rot)
    return pts


S = {}
# --- 正多面体 ---
S["正四面体"] = perms((1, 1, 1))
S["正四面体"] = [p for p in S["正四面体"] if p[0] * p[1] * p[2] > 0]
S["立方体"] = perms((1, 1, 1))
S["正八面体"] = perms((1, 0, 0))
S["正十二面体"] = perms((1, 1, 1)) + perms((0, 1 / phi, phi), even_only=True)
S["正二十面体"] = perms((0, 1, phi), even_only=True)
# --- アルキメデスの立体(六角形以下の面) ---
S["切頂四面体"] = [p for p in perms((3, 1, 1)) if sum(1 for c in p if c < 0) % 2 == 0]
S["立方八面体"] = perms((1, 1, 0))
S["切頂八面体"] = perms((0, 1, 2))
S["斜方立方八面体"] = perms((1, 1, 1 + 2 ** 0.5))
S["二十・十二面体"] = perms((0, 0, phi)) + perms((0.5, phi / 2, phi * phi / 2), even_only=True)
S["切頂二十面体(サッカーボール)"] = (perms((0, 1, 3 * phi), even_only=True) + perms((1, 2 + phi, 2 * phi), even_only=True)
                         + perms((phi, 2, phi ** 3), even_only=True))
S["斜方二十・十二面体"] = (perms((1, 1, phi ** 3), even_only=True) + perms((phi ** 2, phi, 2 * phi), even_only=True)
                   + perms((2 + phi, 0, phi ** 2), even_only=True))
# --- 角柱・反角柱 ---
for n, nm in [(3, "三角柱"), (5, "五角柱"), (6, "六角柱")]:
    S[nm] = stack([(n, 0, 0), (n, 1, 0)])
for n, nm in [(4, "正四角反柱"), (5, "正五角反柱"), (6, "正六角反柱")]:
    S[nm] = stack([(n, 0, 0), (n, anti_h(n), math.pi / n)])
# --- ジョンソンの立体(積み重ねで作れるもの) ---
S["J1 正四角錐"] = stack([(4, 0, 0), ("apex", apex_h(4))])
S["J2 正五角錐"] = stack([(5, 0, 0), ("apex", apex_h(5))])
S["J3 三角キューポラ"] = stack([(6, 0, 0), (3, cupola_h(), math.pi / 6)])
S["J7 正三角錐柱"] = stack([("apex", -apex_h(3)), (3, 0, 0), (3, 1, 0)])
S["J8 正四角錐柱"] = stack([("apex", -apex_h(4)), (4, 0, 0), (4, 1, 0)])
S["J9 正五角錐柱"] = stack([("apex", -apex_h(5)), (5, 0, 0), (5, 1, 0)])
S["J10 ねじれ正四角錐柱"] = stack([("apex", -apex_h(4)), (4, 0, 0), (4, anti_h(4), math.pi / 4)])
S["J11 ねじれ正五角錐柱"] = stack([("apex", -apex_h(5)), (5, 0, 0), (5, anti_h(5), math.pi / 5)])
S["J12 三角両錐"] = stack([("apex", -apex_h(3)), (3, 0, 0), ("apex", apex_h(3))])
S["J13 五角両錐"] = stack([("apex", -apex_h(5)), (5, 0, 0), ("apex", apex_h(5))])
S["J14 正三角両錐柱"] = stack([("apex", -apex_h(3)), (3, 0, 0), (3, 1, 0), ("apex", 1 + apex_h(3))])
S["J15 正四角両錐柱"] = stack([("apex", -apex_h(4)), (4, 0, 0), (4, 1, 0), ("apex", 1 + apex_h(4))])
S["J16 正五角両錐柱"] = stack([("apex", -apex_h(5)), (5, 0, 0), (5, 1, 0), ("apex", 1 + apex_h(5))])
S["J17 ねじれ正四角両錐柱"] = stack([("apex", -apex_h(4)), (4, 0, 0), (4, anti_h(4), math.pi / 4),
                               ("apex", anti_h(4) + apex_h(4))])
S["J18 正三角キューポラ柱"] = stack([(3, -cupola_h(), math.pi / 6), (6, 0, 0), (6, 1, 0)])
S["J22 ねじれ正三角キューポラ柱"] = stack([(3, -cupola_h(), math.pi / 6), (6, 0, 0), (6, anti_h(6), math.pi / 6)])
S["J27 三角オルソバイキューポラ"] = stack([(3, -cupola_h(), math.pi / 6), (6, 0, 0), (3, cupola_h(), math.pi / 6)])
S["J35 正三角オルソバイキューポラ柱"] = stack([(3, -cupola_h(), math.pi / 6), (6, 0, 0), (6, 1, 0), (3, 1 + cupola_h(), math.pi / 6)])
S["J36 正三角ジャイロバイキューポラ柱"] = stack([(3, -cupola_h(), math.pi / 6), (6, 0, 0), (6, 1, 0), (3, 1 + cupola_h(), -math.pi / 6)])
S["J44 ねじれ正三角バイキューポラ柱"] = stack([(3, -cupola_h(), math.pi / 6), (6, 0, 0), (6, anti_h(6), math.pi / 6),
                                    (3, anti_h(6) + cupola_h(), math.pi / 6 + math.pi / 6)])

# --- スナブ(変形)立体 ---
def chiral(vs):
    """偶置換で + の数が偶数のもの + 奇置換で + の数が奇数のもの(スナブ立方体の定義)"""
    out = set()
    ev = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    od = [(0, 2, 1), (2, 1, 0), (1, 0, 2)]
    for p_list, parity in ((ev, 0), (od, 1)):
        for p in p_list:
            base = [vs[i] for i in p]
            for s in itertools.product([1, -1], repeat=3):
                if sum(1 for t in s if t > 0) % 2 == parity:
                    out.add(tuple(round(b * t, 9) for b, t in zip(base, s)))
    return [list(x) for x in out]


trib = (1 + (19 + 3 * 33 ** 0.5) ** (1 / 3) + (19 - 3 * 33 ** 0.5) ** (1 / 3)) / 3
S["変形立方体(スナブ立方体)"] = chiral((1, 1 / trib, trib))


def rotation_group(gens):
    """生成元の回転行列から群を閉包で作る"""
    G = [np.eye(3)]
    frontier = [np.eye(3)]
    while frontier:
        nxt = []
        for g in frontier:
            for h in gens:
                m = h @ g
                if not any(np.allclose(m, x, atol=1e-8) for x in G):
                    G.append(m)
                    nxt.append(m)
        frontier = nxt
    return G


def axis_rot(axis, deg):
    a = np.array(axis, float) / np.linalg.norm(axis)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def snub_dodeca():
    """二十面体回転群(60個)による1点の軌道で、全辺が等長になる点を数値的に探す"""
    from scipy.optimize import minimize
    G = rotation_group([axis_rot((0, 1, phi), 72), axis_rot((1, 1, 1), 120)])
    assert len(G) == 60

    def orbit(p):
        return np.array([g @ p for g in G])

    def cost(q):
        p = np.array([q[0], q[1], 1.0])
        P = orbit(p)
        hull = ConvexHull(P)
        es = set()
        for s in hull.simplices:
            for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
                es.add((min(a, b), max(a, b)))
        L = np.array(sorted(np.linalg.norm(P[a] - P[b]) for a, b in es))
        short = L[:150]  # スナブ十二面体の辺は150本
        return np.var(short / short.mean())

    best = None
    rng = np.random.default_rng(0)
    for _ in range(40):
        r = minimize(cost, rng.uniform(-1, 1, 2), method="Nelder-Mead",
                     options=dict(xatol=1e-12, fatol=1e-16, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    P = orbit(np.array([best.x[0], best.x[1], 1.0]))
    return P.tolist()


S["変形十二面体(スナブ十二面体)"] = snub_dodeca()

S["J62 メタ双欠損二十面体"] = None  # 下で作る
S["J63 三欠損二十面体"] = None

ico = np.array(S["正二十面体"], float)


def diminish(verts, k):
    """二十面体から互いに隣接しない頂点を k 個取り除く(メタ/三欠損)"""
    v = [list(x) for x in verts]
    d = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))
    removed = [v[0]]
    for cand in v:
        if len(removed) == k:
            break
        if all(d(cand, r) > 2.1 and d(cand, r) < 3.5 for r in removed):  # 隣接でも対蹠でもない
            removed.append(cand)
    return [x for x in v if x not in removed]


S["J62 メタ双欠損二十面体"] = diminish(ico, 2)
S["J63 三欠損二十面体"] = diminish(ico, 3)

# --- 八角形・十角形を含む立体 ---
r2 = 2 ** 0.5
S["切頂六面体"] = perms((r2 - 1, 1, 1))
S["切頂立方八面体"] = perms((1, 1 + r2, 1 + 2 * r2))
S["切頂十二面体"] = (perms((0, 1 / phi, 2 + phi), even_only=True) + perms((1 / phi, phi, 2 * phi), even_only=True)
                + perms((phi, 2, phi + 1), even_only=True))
S["切頂二十・十二面体"] = (perms((1 / phi, 1 / phi, 3 + phi), even_only=True) + perms((2 / phi, phi, 1 + 2 * phi), even_only=True)
                     + perms((1 / phi, phi ** 2, 3 * phi - 1), even_only=True) + perms((2 * phi - 1, 2, 2 + phi), even_only=True)
                     + perms((phi, 3, 2 * phi), even_only=True))
for n, nm in [(8, "八角柱"), (10, "十角柱")]:
    S[nm] = stack([(n, 0, 0), (n, 1, 0)])
for n, nm in [(8, "正八角反柱"), (10, "正十角反柱")]:
    S[nm] = stack([(n, 0, 0), (n, anti_h(n), math.pi / n)])


def cupola(n, below=None):
    """n角キューポラ: 上に n 角形、下に 2n 角形 (z=0)。below で下に角柱 / 反角柱をつなぐ"""
    rot = math.pi / (2 * n)
    top = np.array(ring(n, 0, rot))[:, :2]
    bot = np.array(ring(2 * n, 0, 0))[:, :2]
    d = min(np.linalg.norm(t - b) for t in top for b in bot)
    pts = ring(2 * n, 0, 0) + ring(n, math.sqrt(1 - d * d), rot)
    if below == "prism":
        pts += ring(2 * n, -1, 0)
    elif below == "anti":
        pts += ring(2 * n, -anti_h(2 * n), math.pi / (2 * n))
    return pts


def face_normals(pts, k):
    """k 角形の面の外向き単位法線"""
    P = np.array(pts, float)
    hull = ConvexHull(P)
    planes, groups = [], []
    for eq, sx in zip(hull.equations, hull.simplices):
        for i, q in enumerate(planes):
            if np.allclose(eq, q, atol=1e-6):
                groups[i] |= set(sx)
                break
        else:
            planes.append(eq)
            groups.append(set(sx))
    return [q[:3] / np.linalg.norm(q[:3]) for q, g in zip(planes, groups) if len(g) == k]


def rotunda(below=None):
    """五角ロタンダ: 二十・十二面体の半分。赤道の十角形を z=0、角度0に頂点が来るよう回す"""
    P = np.array(S["二十・十二面体"], float)
    z = face_normals(P, 5)[0]
    x = np.cross(z, [1, 0, 0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Q = P @ np.column_stack([x, y, z])
    Q = Q[Q[:, 2] > -1e-9]
    eq = Q[np.abs(Q[:, 2]) < 1e-9]
    a = math.atan2(eq[0, 1], eq[0, 0])
    c, s = math.cos(-a), math.sin(-a)
    Q = Q @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    pts = Q.tolist()
    if below == "prism":
        pts += ring(10, -1, 0)
    elif below == "anti":
        pts += ring(10, -anti_h(10), math.pi / 10)
    return pts


S["J4 四角キューポラ"] = cupola(4)
S["J5 五角キューポラ"] = cupola(5)
S["J6 五角ロタンダ"] = rotunda()
S["J19 正四角キューポラ柱"] = cupola(4, "prism")
S["J20 正五角キューポラ柱"] = cupola(5, "prism")
S["J21 正五角ロタンダ柱"] = rotunda("prism")
S["J23 ねじれ正四角キューポラ柱"] = cupola(4, "anti")
S["J24 ねじれ正五角キューポラ柱"] = cupola(5, "anti")
S["J25 ねじれ正五角ロタンダ柱"] = rotunda("anti")


def diminish_rid(k):
    """斜方二十・十二面体から五角形の面を k 枚(互いに対蹠 / 離れた位置)ぶん頂点ごと取り除く"""
    P = np.array(S["斜方二十・十二面体"], float)
    n0 = face_normals(P, 5)[0]
    chosen = [n0]
    if k >= 2:
        chosen.append(-n0)
    keep = np.ones(len(P), bool)
    for a in chosen:
        dots = P @ a
        keep &= dots < dots.max() - 1e-6
    return P[keep].tolist()


S["J76 欠損斜方二十・十二面体"] = diminish_rid(1)
S["J80 パラ双欠損斜方二十・十二面体"] = diminish_rid(2)


# 板どうしがぶつかって組めないもの (check_assembly.py で確認)
NG = {"J2 正五角錐", "J4 四角キューポラ", "J5 五角キューポラ"}


def analyze(pts):
    P = np.array(pts, float)
    hull = ConvexHull(P)
    # 共面の三角形を面にまとめる
    planes = []
    simp_face = []
    for eq in hull.equations:
        for i, q in enumerate(planes):
            if np.allclose(eq, q, atol=1e-6):
                simp_face.append(i)
                break
        else:
            planes.append(eq)
            simp_face.append(len(planes) - 1)
    faces = {i: set() for i in range(len(planes))}
    for s, f in zip(hull.simplices, simp_face):
        faces[f] |= set(s)
    edge_len = min(np.linalg.norm(P[a] - P[b]) for a, b in itertools.combinations(range(len(P)), 2))
    counts = {}
    for f, vs in faces.items():
        counts[len(vs)] = counts.get(len(vs), 0) + 1
    # 隣接面(2頂点共有)の二面角
    dih = {}
    for i, j in itertools.combinations(faces, 2):
        if len(faces[i] & faces[j]) == 2:
            a, b = sorted((len(faces[i]), len(faces[j])))
            ang = 180 - math.degrees(math.acos(np.clip(np.dot(planes[i][:3], planes[j][:3]), -1, 1)))
            dih.setdefault((a, b), set()).add(round(ang, 2))
    # 辺長が揃っているか
    lens = set()
    for f, vs in faces.items():
        pass
    return counts, dih, edge_len


def main():
  res = {}
  for nm, pts in S.items():
      counts, dih, el = analyze(pts)
      ok = all(k in (3, 4, 5, 6, 8, 10) for k in counts)
      res[nm] = dict(faces={str(k): v for k, v in sorted(counts.items())}, total=sum(counts.values()),
                     dihedral={f"{a}-{b}": sorted(v) for (a, b), v in sorted(dih.items())}, regular=ok)
      print(f"{nm:28s} 面 {dict(sorted(counts.items()))} 計{sum(counts.values()):3d}  "
            + "  ".join(f"{a}-{b}:{sorted(v)}" for (a, b), v in sorted(dih.items())))
  json.dump(res, open("poly_list.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
