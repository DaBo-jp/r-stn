import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 設定エリア ===
TARGET_FREQ = 20.0   # 抽出したい周波数 (学習したパスを見たいなら20.0)
FREQ_TOLERANCE = 5.0 # 許容誤差 (±5Hz)
SLICE_AXIS = 1       # 0:Z軸, 1:Y軸(推奨), 2:X軸 で切断
# =================

def visualize_analysis(file_path):
    if not os.path.exists(file_path): return
    print(f"Analyzing {file_path} ...")
    
    try:
        loader = np.load(file_path)
        freqs = loader['freqs'] 
        amps = loader['amps']
        times = loader['compute_times']
        name = str(loader['name'])
        size = int(loader['size'])
        interval = int(loader['save_interval']) if 'save_interval' in loader else 1
    except Exception as e:
        print(f"Error: {e}")
        return

    frames = freqs.shape[0]
    
    # 座標生成
    coords = np.zeros((size**3, 3))
    for z in range(size):
        for y in range(size):
            for x in range(size):
                idx = x + y*size + z*size*size
                coords[idx] = [z, y, x]

    # 出力フォルダ作成
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        
    out_dir = os.path.join(reports_dir, f"analysis_{name}")
    os.makedirs(out_dir, exist_ok=True)

    def save_multi_view(idx):
        real_step = idx * interval
        current_amps = amps[idx]
        current_freqs = freqs[idx]
        
        # 3つのプロットを作成 (横に並べる)
        fig = plt.figure(figsize=(18, 6))
        
        # --- View 1: Frequency Filter (特定周波数のパス抽出) ---
        ax1 = fig.add_subplot(131, projection='3d')
        # TARGET_FREQ に近いノードだけ抽出
        freq_mask = np.abs(current_freqs - TARGET_FREQ) < FREQ_TOLERANCE
        amp_mask = current_amps > 1.0 # ある程度活動しているもの
        mask1 = freq_mask & amp_mask
        
        if np.any(mask1):
            c = coords[mask1]
            f = current_freqs[mask1]
            a = current_amps[mask1]
            ax1.scatter(c[:,2], c[:,1], c[:,0], c=f, s=a*0.5, 
                       cmap='coolwarm', vmin=-50, vmax=50, alpha=0.8)
        ax1.set_title(f"Target: {TARGET_FREQ}Hz (±{FREQ_TOLERANCE})")
        
        # --- View 2: Cross Section (断面図) ---
        ax2 = fig.add_subplot(132, projection='3d')
        # 中心付近の断面のみ抽出
        mid = size // 2
        slice_mask = (coords[:, SLICE_AXIS] >= mid-1) & (coords[:, SLICE_AXIS] <= mid+1)
        mask2 = slice_mask & (current_amps > 1.0)
        
        if np.any(mask2):
            c = coords[mask2]
            f = current_freqs[mask2]
            a = current_amps[mask2]
            ax2.scatter(c[:,2], c[:,1], c[:,0], c=f, s=a*2.0, 
                       cmap='coolwarm', vmin=-50, vmax=50, alpha=0.9)
        ax2.set_title(f"Cross Section (Axis {SLICE_AXIS})")

        # --- View 3: High Amplitude (高エネルギーノード) ---
        ax3 = fig.add_subplot(133, projection='3d')
        # 振幅が大きい上位ノードのみ
        mask3 = current_amps > 80.0
        
        if np.any(mask3):
            c = coords[mask3]
            f = current_freqs[mask3]
            a = current_amps[mask3]
            # ここは透明度を下げて内部を見えやすく
            ax3.scatter(c[:,2], c[:,1], c[:,0], c=f, s=a*0.3, 
                       cmap='coolwarm', vmin=-50, vmax=50, alpha=0.3)
        ax3.set_title("High Amplitude (>80.0)")

        # 共通設定
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(-1, size); ax.set_ylim(-1, size); ax.set_zlim(-1, size)
            ax.set_facecolor('black')
            ax.grid(False)
            ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
            ax.view_init(elev=20, azim=45)

        fig.patch.set_facecolor('black')
        plt.suptitle(f"{name} | Step: {real_step} | Time: {times[idx]:.2f}s", color='white', fontsize=16)
        
        out_path = os.path.join(out_dir, f"step{real_step:03d}.png")
        plt.savefig(out_path, facecolor='black', bbox_inches='tight')
        plt.close()
        print(f"  > Generated: {out_path}")

    # 生成対象: 最初、学習完了直前、推論時
    # indexベースで計算 (step 200 付近を見たい)
    target_steps = [0, 100, 200, 300, 399] # 見たいステップ(実数)
    
    for t in target_steps:
        # 実ステップ数からindexへ変換
        idx = t // interval
        if idx < frames:
            save_multi_view(idx)

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    target_dir = os.path.join(BASE_DIR, "..", "data", "cpp_output")

    

    if os.path.exists(target_dir):
        for f in sorted(os.listdir(target_dir)):
            if f.endswith(".npz"):
                visualize_analysis(os.path.join(target_dir, f))