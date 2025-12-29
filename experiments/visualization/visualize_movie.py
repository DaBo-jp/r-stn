import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg') # 画面表示せずバックグラウンドで描画
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# === 設定 ===
TARGET_FREQ = 20.0    # 可視化したいターゲット周波数
FREQ_TOLERANCE = 8.0  # 許容誤差 (±8Hz)
SLICE_AXIS = 1        # 断面を切る軸 (0:Z, 1:Y, 2:X)
SKIP_FRAME = 1        # フレームを間引く場合 (1=全フレーム, 2=半分...)
FPS = 30              # 動画のフレームレート
# ============

def create_movie(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Creating movie for {file_path} ...")
    try:
        loader = np.load(file_path)
        freqs = loader['freqs'] 
        amps = loader['amps']
        times = loader['compute_times']
        name = str(loader['name'])
        size = int(loader['size'])
        interval = int(loader['save_interval']) if 'save_interval' in loader else 1
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return
    
    total_frames = freqs.shape[0]
    
    # 座標生成
    coords = np.zeros((size**3, 3))
    for z in range(size):
        for y in range(size):
            for x in range(size):
                idx = x + y*size + z*size*size
                coords[idx] = [z, y, x] # Z(奥行), Y(縦), X(横)

    # 動画用フィギュアの準備 (2画面横並び)
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 背景色設定
    fig.patch.set_facecolor('black')

    # フレーム更新関数
    def update(frame_idx):
        real_step = frame_idx * interval
        current_amps = amps[frame_idx]
        current_freqs = freqs[frame_idx]
        time_val = times[frame_idx] if frame_idx < len(times) else 0.0

        ax1.clear()
        ax2.clear()
        
        # 共通設定
        for ax in [ax1, ax2]:
            ax.set_xlim(-1, size); ax.set_ylim(-1, size); ax.set_zlim(-1, size)
            ax.set_facecolor('black')
            ax.grid(False)
            ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
            # 視点調整 (見やすい角度へ)
            ax.view_init(elev=25, azim=-60)

        # --- 左画面: Target Path (特定周波数抽出) ---
        # TARGET_FREQ に近く、かつ活動しているノードのみ抽出
        freq_mask = np.abs(current_freqs - TARGET_FREQ) < FREQ_TOLERANCE
        amp_mask = current_amps > 2.0 # ノイズカット閾値
        mask1 = freq_mask & amp_mask
        
        if np.any(mask1):
            c = coords[mask1]
            f = current_freqs[mask1]
            a = current_amps[mask1]
            # 振幅をサイズに反映。Case6の冷却時は振幅が小さくなるので見やすくなる。
            ax1.scatter(c[:,2], c[:,1], c[:,0], c=f, s=a*0.8, 
                       cmap='coolwarm', vmin=-50, vmax=50, alpha=0.8, edgecolors='none')
        
        ax1.set_title(f"Target Path ({TARGET_FREQ}Hz ±{FREQ_TOLERANCE})", color='white')
        ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z (Depth)")

        # --- 右画面: Cross Section (断面図) ---
        # 指定軸の中央付近の断面を抽出
        mid = size // 2
        # 断面の厚みを持たせる (前後1ノード分)
        slice_mask = (coords[:, SLICE_AXIS] >= mid-1) & (coords[:, SLICE_AXIS] <= mid+1)
        mask2 = slice_mask & (current_amps > 1.0)
        
        if np.any(mask2):
            c = coords[mask2]
            f = current_freqs[mask2]
            a = current_amps[mask2]
            ax2.scatter(c[:,2], c[:,1], c[:,0], c=f, s=a*2.5, 
                       cmap='coolwarm', vmin=-50, vmax=50, alpha=0.9, edgecolors='none')

        axis_name = ["Z", "Y", "X"][SLICE_AXIS]
        ax2.set_title(f"Cross Section ({axis_name}-Axis Center)", color='white')
        
        # --- 全体情報表示 (HUD) ---
        info_text = f"Step: {real_step:04d}\nTime: {time_val:.2f}s"
        # 便宜上 ax2 にテキストを置く
        ax2.text2D(0.02, 0.95, info_text, transform=ax2.transAxes, 
                   color='yellow', fontsize=14, family='monospace',
                   bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

        # 進捗表示
        sys.stdout.write(f"\rProcessing frame {frame_idx}/{total_frames}...")
        sys.stdout.flush()

    # 動画生成実行
    # frames=range(...) で間引きを制御
    frame_iter = range(0, total_frames, SKIP_FRAME)
    anim = animation.FuncAnimation(fig, update, frames=frame_iter, interval=1000//FPS)

    # 保存処理
    out_base = os.path.splitext(os.path.basename(file_path))[0]
    
    # FFmpegが使えるか試す (MP4出力優先)
    try:
        out_mp4 = f"{out_base}.mp4"
        print(f"\nSaving to {out_mp4} (High Quality)...")
        # bitrateを上げると画質向上
        anim.save(out_mp4, writer='ffmpeg', fps=FPS, dpi=100, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-b:v', '5000k'])
        print("Done!")
    except Exception as e:
        print(f"\nFFmpeg not found or failed ({e}). Falling back to GIF.")
        out_gif = f"{out_base}.gif"
        print(f"Saving to {out_gif} (may take time)...")
        # GIFは容量削減のためFPSを落とす
        anim.save(out_gif, writer='pillow', fps=15, dpi=80)
        print("Done!")

    plt.close()

if __name__ == "__main__":
    # Case 5 or 6 のデータを探す
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(BASE_DIR, "..", "data", "cpp_output")

    target_file = None
    if os.path.exists(target_dir):
        
    targets = ["Case5_Dynamic.npz", "Case6_Discrete.npz"]
    
    found = False
    for t in targets:
        path = os.path.join(target_dir, t)
        if os.path.exists(path):
            create_movie(path)
            found = True
            
    if not found:
        print("No target data (Case5/Case6) found in data_cpp/.")
        print("Run 'run_cpp_sim.py' first.")