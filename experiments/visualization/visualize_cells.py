import numpy as np
import matplotlib
matplotlib.use('Agg')  # ★これを追加（GUIを使わずに描画する設定）
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
import re
import math

# =========================================================================
# 設定
# =========================================================================
DATA_DIR = "experiment_data"
OUTPUT_VIDEO = "brain_wave_comparison_7x7.mp4"
N = 32
GRID_SIZE = 7  # 7x7 = 49個
MAX_FILES = GRID_SIZE * GRID_SIZE

def get_target_pos(s, steps, size):
    cy, cx = size // 2, size // 2
    radius = size * 0.35
    angle = (s / float(steps)) * 2 * math.pi 
    ty = cy + int(radius * math.sin(angle))
    tx = cx + int(radius * math.cos(angle))
    return max(0, min(size-1, tx)), max(0, min(size-1, ty))

# =========================================================================
# 1. ファイル収集 & ソート (高粘性・高減衰を優先)
# =========================================================================
files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
if not files:
    print("ファイルが見つかりません。")
    exit()

def get_params(f):
    v = float(re.search(r"Visc([\d\.]+)", f).group(1))
    a = float(re.search(r"Attn([\d\.]+)", f).group(1))
    i = float(re.search(r"Inert([\d\.]+)", f).group(1))
    return v, a, i

# 高粘性・高減衰の「エリート」を優先して49個選ぶ
files.sort(key=lambda f: (get_params(f)[0], get_params(f)[1]), reverse=True)
selected_files = files[:MAX_FILES]
num_files = len(selected_files)

# タイルの縦横を計算
rows = math.ceil(num_files / GRID_SIZE)
cols = min(num_files, GRID_SIZE)

# =========================================================================
# 2. データのプリロード
# =========================================================================
all_amps = []
titles = []
for f in selected_files:
    data = np.load(f)
    amp_raw = data['amps']  # (Steps, N*N) または (Steps, N*N*N)
    
    # --- ここでリシェイプ処理を追加 ---
    num_steps = amp_raw.shape[0]
    elements_per_step = amp_raw.shape[1] if amp_raw.ndim > 1 else amp_raw.size // num_steps

    if elements_per_step == N * N:
        # 2Dデータの場合
        amp = amp_raw.reshape(num_steps, N, N)
    elif elements_per_step == N * N * N:
        # 3Dデータの場合 (Depth, H, W) に戻して、最大値投影(Max Projection)
        amp_3d = amp_raw.reshape(num_steps, N, N, N)
        amp = np.max(amp_3d, axis=1) 
    else:
        print(f"警告: {f} のデータサイズが不正です。スキップします。")
        continue
    # --------------------------------

    all_amps.append(amp)
    v, a, i = get_params(f)
    titles.append(f"V:{v:.2f} A:{a:.1f}")

if not all_amps:
    print("有効なデータが読み込めませんでした。")
    exit()

steps = all_amps[0].shape[0]

# =========================================================================
# 3. アニメーション作成
# =========================================================================
fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2.2), constrained_layout=True)
fig.patch.set_facecolor('black')
axes = np.array(axes).flatten()

ims = []
tgt_dots = []
pk_dots = []

for i in range(len(axes)):
    if i < num_files:
        ax = axes[i]
        # 背景（波）
        im = ax.imshow(all_amps[i][0], cmap='magma', origin='lower', extent=[0, N, 0, N], vmin=0, vmax=20)
        # ターゲット（赤点）
        tgt, = ax.plot([], [], 'ro', markersize=4)
        # ピーク（白点）
        pk, = ax.plot([], [], 'wx', markersize=4)
        
        ax.set_title(titles[i], color='white', fontsize=8)
        ax.axis('off')
        
        ims.append(im)
        tgt_dots.append(tgt)
        pk_dots.append(pk)
    else:
        axes[i].axis('off')

def update(s):
    tx, ty = get_target_pos(s, steps, N)
    changed_artists = []
    
    for i in range(num_files):
        # 波の更新
        ims[i].set_array(all_amps[i][s])
        # ターゲット位置
        tgt_dots[i].set_data([tx], [ty])
        # 自律ピーク位置
        grid = all_amps[i][s]
        peak_idx = np.argmax(grid)
        py, px = np.unravel_index(peak_idx, (N, N))
        pk_dots[i].set_data([px], [py])
        
        changed_artists.extend([ims[i], tgt_dots[i], pk_dots[i]])
    
    return changed_artists

ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=True)

# 保存
print(f"🎬 {num_files}個の個体をタイル状に並べて動画を生成中...")
ani.save(OUTPUT_VIDEO, writer='ffmpeg', fps=20, dpi=100)
print(f"✅ 保存完了: {OUTPUT_VIDEO}")