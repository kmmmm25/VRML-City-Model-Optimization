import numpy as np
import math
from collections import defaultdict
from scipy.spatial import Delaunay, cKDTree

# =============================
# 入力
# =============================
INPUT_FILE = "53394640_dsm_1m.dat"


def load_xyz(path=INPUT_FILE):
    pts = []
    with open(path, "r") as f:
        for line in f:
            sp = line.split()
            if len(sp) < 3:
                continue
            try:
                pts.append((float(sp[0]), float(sp[1]), float(sp[2])))
            except ValueError:
                pass
    return np.array(pts, dtype=float)


# =============================
# 欠損点補完（z<0 / NaN）
# =============================
def fill_missing_points(points):
    points = points.copy()

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    z = np.where(z < 0, np.nan, z)
    points[:, 2] = z

    nan_mask = np.isnan(z)
    print(f"欠損点数（補完前）: {np.sum(nan_mask)}")

    unique_x = np.unique(x)
    unique_y = np.unique(y)
    width = len(unique_x)
    height = len(unique_y)

    def get_neighbors_8(i):
        neighbors = []
        row = i // width
        col = i % width
        deltas = [
            (-1, -1), (-1,  0), (-1,  1),
            ( 0, -1),          ( 0,  1),
            ( 1, -1), ( 1,  0), ( 1,  1)
        ]
        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width:
                neighbors.append(nr * width + nc)
        return neighbors

    def fill_one(i):
        neighbors = get_neighbors_8(i)
        zs = []
        for k in neighbors:
            if not np.isnan(points[k, 2]) and points[k, 2] >= 0:
                zs.append(points[k, 2])

        if len(zs) == 0:
            fill_z = 0.0
        else:
            diff = max(zs) - min(zs)
            threshold = 4.0  # 建物高さ保持寄り
            if diff < threshold:
                fill_z = np.mean(zs)
            else:
                zs_sorted = sorted(zs)
                med = zs_sorted[len(zs_sorted)//2]
                low = [v for v in zs if v <= med]
                high = [v for v in zs if v > med]
                fill_z = np.mean(low) if len(low) >= len(high) else np.mean(high)

        points[i, 2] = fill_z

    for i in range(len(points)):
        if np.isnan(points[i, 2]) or points[i, 2] < 0:
            fill_one(i)

    print(f"欠損点数（補完後）: {np.sum(np.isnan(points[:,2]))}")
    return points
# =============================
#  地面判定１（仮平面作成）
# =============================
def kariheimenn(filtered_points):
    N = 1 #　θ設定
    ansans_y = []
    ansans_x = []
    y_target = filtered_points[0][1]
    for _ in range(0,1131,10):
        x_line,z_line = [],[]
        # --- y軸で線形的に抽出したい範囲を指定 ---
        y_target += 10
        #print(y_target)
        # y軸範囲でフィルタリング
        for j in range(0,len(filtered_points)):
            if filtered_points[j][1] == y_target:
                x_line.append(filtered_points[j][0])
                z_line.append(filtered_points[j][2])
        print("y",y_target,len(x_line),len(z_line))
        ans1,ans2 = [[-1 for _ in range(len(z_line))] for _ in range(0,3)],[[-1 for _ in range(len(z_line))] for _ in range(0,3)]
        a1_deg,a2_deg = [1000] * len(z_line),[1000] * len(z_line)
        j = 0
        for i in range(0,len(z_line)):
            base,dz_val = abs(i - j),z_line[i] - z_line[j]
            if base == 0:
                continue  # 0除算を避ける（または特別扱い）
            a2_deg[i] = dz_val/base
            dz_val = abs(dz_val)
            hypotenuse = math.sqrt(dz_val*dz_val + base*base)

            cos_theta = base / hypotenuse # cosθ = 隣辺 / 斜辺 = base / hypotenuse

            cos_theta = np.clip(cos_theta, -1.0, 1.0) # 浮動小数誤差対策で -1〜1 にクリップ

            angle_deg = np.degrees(np.arccos(cos_theta))

            if angle_deg <= N:
                if abs(j-i)>1:
                    for k in range(j+1,i):
                        ans2[0][k],ans2[1][k],ans2[2][k] = x_line[i],y_target,z_line[j]+(z_line[i]-z_line[j]*abs(k-j)/abs(i-j))
                ans2[0][i],ans2[1][i],ans2[2][i] = x_line[i],y_target, z_line[i]
                j = i

        j = len(z_line)-1
        for i in range(len(z_line)-2,0,-1):
            base,dz_val = abs(i - j),z_line[i] - z_line[j]
            if base == 0:
                continue  # 0除算を避ける（または特別扱い）
            a1_deg[i] = dz_val/base
            dz_val = abs(dz_val)
            hypotenuse = math.sqrt(dz_val*dz_val + base*base)

            cos_theta = np.clip(cos_theta, -1.0, 1.0) # 浮動小数誤差対策で -1〜1 にクリップ

            angle_deg = np.degrees(np.arccos(cos_theta))

            if angle_deg <= N:
                if abs(j-i)>1:
                    for k in range(j+1,i):
                        ans1[0][k],ans1[1][k],ans1[2][k] = x_line[i],y_target,z_line[j]+(z_line[i]-z_line[j]*abs(k-j)/abs(i-j))
                ans1[0][i],ans1[1][i],ans1[2][i] = x_line[i], y_target, z_line[i]
                j = i

        i,j,k = 0,0,0
        ansans = []
        for i in range(0,len(z_line)):
            num,n = 900,max(ans1[2][i],ans2[2][i])
            if a1_deg[i] >= -1  and ans1[2][i] != -1:
                num,a = a1_deg[i],ans1[2][i]
                #print("a 1",a)
            if a2_deg[i] < num and ans2[2][i] != -1:
                num,a = a2_deg[i],ans2[2][i]
                #print("a 2",a)
            if  n > -1:
                x,y = ans1[0][i],ans1[1][i],
                if ans2[0][i] != -1:
                    x,y = ans2[0][i],ans2[1][i]
                ansans.append([x,y,a])
        ansans.append([x,y,a])
        ansans_y.append(ansans)
    for i in range(len(ansans_y)):
        print(len(ansans_y[i]))
    print("ansans_y:", len(ansans_y), "lines")
    x_target = filtered_points[0][0]
    for i in range(0,925, 10):
        y_line,z_line = [],[]
        # --- x軸で線形的に抽出したい範囲を指定 ---
        x_target += 10
        # x軸範囲でフィルタリング
        for j in range(0,len(filtered_points)):
            if filtered_points[j][0] == x_target:
                y_line.append(filtered_points[j][1])
                z_line.append(filtered_points[j][2])
        print("x",x_target,len(y_line),len(z_line))
        ans1, ans2 = [[-1 for _ in range(len(z_line))] for _ in range(3)],[[-1 for _ in range(len(z_line))] for _ in range(3)]
        a1_deg = [1000] * len(z_line)
        a2_deg = [1000] * len(z_line)

        # --- 正方向 ---
        j = 0
        for i in range(0, len(z_line)):
            base = abs(i - j)
            if base == 0:
                continue

            dz_val = z_line[i] - z_line[j]
            a2_deg[i] = dz_val / base

            hypotenuse = math.sqrt(dz_val * dz_val + base * base)
            angle_deg = np.degrees(np.arctan2(abs(dz_val), base))
            if angle_deg <= N:
                if abs(j-i)>1:
                    for k in range(j+1,i):
                        ans2[0][k],ans2[1][k],ans2[2][k] = x_target, y_line[i],z_line[j]+(z_line[i]-z_line[j]*abs(k-j)/abs(i-j))
                ans2[0][i], ans2[1][i], ans2[2][i] = x_target, y_line[i], z_line[i]
                j = i

        # --- 逆方向 ---
        j = len(z_line) - 1
        for i in range(len(z_line) - 1, 0, -1):
            base = abs(i - j)
            if base == 0:
                continue

            dz_val = z_line[i] - z_line[j]
            a1_deg[i] = dz_val / base

            hypotenuse = math.sqrt(dz_val * dz_val + base * base)
            cos_theta = np.clip(base / hypotenuse, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_theta))

            if angle_deg <= N:
                if abs(j-i)>1:
                    for k in range(j+1,i):
                        ans1[0][k],ans1[1][k],ans1[2][k] = x_target, y_line[i],z_line[j]+(z_line[i]-z_line[j]*abs(k-j)/abs(i-j))
                ans1[0][i], ans1[1][i], ans1[2][i] = x_target, y_line[i], z_line[i]
                j = i

        # --- 統合 ---
        ansans = []
        for i in range(len(z_line)):
            num = 900
            n = max(ans1[2][i], ans2[2][i])

            if a1_deg[i] >= -1 and ans1[2][i] != -1:
                num, a = a1_deg[i], ans1[2][i]

            if a2_deg[i] < num and ans2[2][i] != -1:
                num, a = a2_deg[i], ans2[2][i]

            if n > -1:
                x_val, y_val = ans1[0][i], ans1[1][i]
                if ans2[0][i] != -1:
                    x_val, y_val = ans2[0][i], ans2[1][i]
                ansans.append([x_val, y_val, a])
        ansans_x.append(ansans)
    for i in range(len(ansans_x)):
        print(len(ansans_x[i]))
    print("ansans_x:", len(ansans_x), "lines")
    return ansans_x,ansans_y
# =============================
# ① タイル分割 & zrange
# =============================
def compute_stats_per_tile(xyz, nx=8, ny=8):
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    tiles_xyz = [[[] for _ in range(nx)] for _ in range(ny)]
    tiles_z = [[[] for _ in range(nx)] for _ in range(ny)]

    for x, y, z in xyz:
        ix = min(int((x - x_min) / dx), nx - 1)
        iy = min(int((y - y_min) / dy), ny - 1)
        tiles_xyz[iy][ix].append((x, y, z))
        tiles_z[iy][ix].append(z)

    zrange_map = np.zeros((ny, nx))
    for iy in range(ny):
        for ix in range(nx):
            zs = tiles_z[iy][ix]
            if zs:
                zrange_map[iy, ix] = max(zs) - min(zs)

    return zrange_map, tiles_xyz


# =============================
# ③ g1 多段判定
# =============================
def decide_grid_size(k):
    if k > 8.0:
        return 2.0
    elif k > 4.0:
        return 4.0
    elif k > 2.0:
        return 10.0
    else:
        return 30.0


# =============================
# ⑤ グリッド削減 + 側面代表点
# =============================
def reduce_points_in_tile(points, grid_size, g2):
    if len(points) == 0:
        return [], []

    pts = np.array(points)
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    gx = ((pts[:, 0] - x0) / grid_size).astype(int)
    gy = ((pts[:, 1] - y0) / grid_size).astype(int)

    grid = defaultdict(list)
    for ix, iy, x, y, z in zip(gx, gy, pts[:, 0], pts[:, 1], pts[:, 2]):
        grid[(ix, iy)].append((x, y, z))

    reduced = []
    wall_reps = []

    for cells in grid.values():
        if len(cells) < 3:
            reduced.extend(cells)
            continue

        zs = np.array([p[2] for p in cells])
        zrange = zs.max() - zs.min()

        if zrange < g2 * 1.5:
            reduced.append(cells[np.argmax(zs)])
        else:
            reduced.extend(cells)

        wall_reps.append(cells[np.argmax(zs)])

    return reduced, wall_reps

# =============================
# ⑥ 全体削減
# =============================
def adaptive_reduce_xyz(xyz, nx=8, ny=8):
    zrange_map, tiles_xyz = compute_stats_per_tile(xyz, nx, ny)

    reduced_all = []
    wall_points = []

    for iy in range(ny):
        for ix in range(nx):
            orig = tiles_xyz[iy][ix]

            k = zrange_map[iy, ix]
            grid_size = decide_grid_size(k)

            if grid_size >= 12:
                g2 = 20.0
            elif grid_size >= 8:
                g2 = 12.0
            else:
                g2 = 6.0

            reduced, wall = reduce_points_in_tile(orig, grid_size, g2)
            reduced_all.extend(reduced)

            if k > 4.0 and grid_size <= 4.0:
                wall_points.extend(wall)

    return np.array(reduced_all), np.array(wall_points)

# =============================
# 地面判定２（閾値による仮地面との差を見る）
# =============================
def judge_ground(points, ansans_x, ansans_y):
    x, y, z = points[0],points[1],points[2]
    r_x = -7539.00
    r_y = -33277.00
    # --- x軸方向の参照 ---
    d = int(abs(r_y-y))
    d_t,d_o = int(d//10),int(d%10)

    if d_t+1 >= len(ansans_y) and d >= len(ansans_y[d_t]):
        z_y = ansans_y[d_t][d][2]
        print(d,len(ansans_y[0]),ansans_y[d_t][d][2])
    else:
        z_y = ansans_y[d_t][d][2] + (ansans_y[d_t][d][2] - ansans_y[d_t+1][d][2])*d_o/10

        # --- y軸方向の参照 ---
    p = int(abs(r_x-x))
    d_f,d_h = int(p//10),int(p%10)

    if d_f+1 >= len(ansans_x) and p >= len(ansans_x[d_f]):
        print("y",p,len(ansans_x[d_f]),ansans_x[d_f][p][2])
        z_y = ansans_x[d_f][p][2]
    else:
        z_x = ansans_x[d_f][p][2] + (ansans_x[d_f][p][2] - ansans_x[d_f+1][p][2])*d_h/10
    
    if z >= min(z_y,z_x)+2.0 and z <= min(z_y,z_x)+2.0:
        return True
    else:
        return False

# =============================
# ⑦ 側面エッジ検出
# =============================
def detect_vertical_edges(points, dz_min=6.0):
    if len(points) < 2:
        return []

    tree = cKDTree(points[:, :2])
    edges = []

    for i, p in enumerate(points):
        _, idx = tree.query(p[:2], k=2)
        j = idx[1]
        if abs(p[2] - points[j][2]) > dz_min:
            edges.append((i, j))

    return edges
# =============================
# ② 平滑化
# =============================
def smooth_map(m):
    ny, nx = m.shape
    out = np.zeros_like(m)
    for iy in range(ny):
        for ix in range(nx):
            vals = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = iy + dy, ix + dx
                    if 0 <= yy < ny and 0 <= xx < nx:
                        vals.append(m[yy, xx])
            out[iy, ix] = np.mean(vals)
    return out

# =============================
# ⑧ WRL 出力
# =============================

from scipy.spatial import Delaunay

from scipy.spatial import Delaunay

def f(v):
    return f"{v:.5f}"

def write_wrl(points, fjimen_x, fjimen_y, out_path="output.wrl"):
    tri = Delaunay(points[:, :2])

    # 三角形ごとの判定部分（必要なら何かに使う）
    face_colors = []
    for a, b, c in tri.simplices:
        pa, pb, pc = points[a], points[b], points[c]
        if (judge_ground(pa, fjimen_x, fjimen_y) and
            judge_ground(pb, fjimen_x, fjimen_y) and
            judge_ground(pc, fjimen_x, fjimen_y)):
            face_colors.append([0.0, 1.0, 0.0])  # 地面
        else:
            face_colors.append([1.0, 0.0, 0.0])  # 非地面

    # 頂点ごとの色を決定（頂点単位でjudge_ground）
    vertex_colors = {}
    for i, p in enumerate(points):
        if judge_ground(p, fjimen_x, fjimen_y):
            vertex_colors[i] = (0.0, 1.0, 0.0)  # 緑
        else:
            vertex_colors[i] = (1.0, 0.0, 0.0)  # 赤

    colors = [vertex_colors[i] for i in range(len(points))]

    with open(out_path, "w", encoding="utf-8") as w:
        w.write("#VRML V2.0 utf8\n")
        w.write("Shape {\n")
        w.write("  appearance Appearance { material Material { diffuseColor 0.8 0.8 0.8 } }\n")
        w.write("  geometry IndexedFaceSet {\n")
        w.write("    coord Coordinate { point [\n")
        for x, y, z in points:
            w.write(f"      {f(x)} {f(y)} {f(z)},\n")
        w.write("    ] }\n")
        w.write("    color Color { color [\n")
        for r, g, b in colors:
            w.write(f"      {f(r)} {f(g)} {f(b)},\n")
        w.write("    ] }\n")
        w.write("    colorPerVertex TRUE\n")
        w.write("    coordIndex [\n")
        for a, b, c in tri.simplices:
            w.write(f"      {a}, {b}, {c}, -1,\n")
        w.write("    ]\n")
        w.write("  }\n")
        w.write("}\n")

    print(f"WRLファイルを書き出しました: {out_path}")




# =============================
# 実行
# =============================
if __name__ == "__main__":
    xyz = load_xyz()
    xyz = fill_missing_points(xyz)
    fjimen_x,fjimen_y = kariheimenn(xyz)
    reduced_xyz, wall_pts = adaptive_reduce_xyz(xyz)
    print("元の点数:", len(xyz))
    print("削減後点数:", len(reduced_xyz))
    print(f"削減率: {(1 - len(reduced_xyz)/len(xyz))*100:.2f}%")

    # ★ 最後にWRL出力
    write_wrl(reduced_xyz,fjimen_x,fjimen_y,"final_output.wrl")

