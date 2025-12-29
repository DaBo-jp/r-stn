import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(file_path):
    if not os.path.exists(file_path): 
        print(f"File not found: {file_path}")
        return

    print(f"Processing {file_path} ...")
    try:
        loader = np.load(file_path)
        freqs = loader['freqs'] 
        amps = loader['amps']
        fats = loader['fats']
        times = loader['compute_times']
        name = str(loader['name'])
        size = int(loader['size'])
        # 間引き間隔を取得（なければ1とみなす）
        interval = int(loader['save_interval']) if 'save_interval' in loader else 1
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return
    
    frames = freqs.shape[0]
    
    # 座標生成 (32^3 程度なら生成しても一瞬)
    coords = np.zeros((size**3, 3))
    for z in range(size):
        for y in range(size):
            for x in range(size):
                idx = x + y*size + z*size*size
                coords[idx] = [z, y, x]

    fig = plt.figure(figsize=(14, 10)) # 横幅を広げて情報スペース確保
    ax = fig.add_subplot(111, projection='3d')

    def save_frame(idx):
        ax.clear()
        
        # 実際のステップ数
        real_step = idx * interval
        
        # --- 統計計算 (Python側で再計算) ---
        current_amps = amps[idx]
        current_fats = fats[idx]
        current_freqs = freqs[idx]
        
        active_count = np.count_nonzero(current_amps > 1.0)
        max_amp = np.max(current_amps)
        avg_fat = np.mean(current_fats)
        active_ratio = (active_count / (size**3)) * 100.0
        
        # --- 描画 (軽量化のため閾値フィルタ) ---
        mask = current_amps > 5.0
        if np.any(mask):
            c_d = coords[mask]
            f_d = current_freqs[mask]
            a_d = current_amps[mask]
            
            ax.scatter(c_d[:, 2], c_d[:, 1], c_d[:, 0],
                       c=f_d, s=a_d * 1.5,
                       cmap='coolwarm', vmin=-50, vmax=50,
                       edgecolors='none', alpha=0.7)
        
        # --- HUD (Head-Up Display) 情報表示 ---
        # 左上: 基本情報
        info_text = (
            f"Step: {real_step}\n"
            f"Time: {times[idx]:.4f} s\n"
            f"Mode: {'LEARNING' if real_step < 200 else 'INFERENCE'}"
        )
        ax.text2D(0.02, 0.95, info_text, transform=ax.transAxes, 
                  color='white', fontsize=14, weight='bold',
                  bbox=dict(facecolor='black', alpha=0.7, edgecolor='gray'))

        # 右上: 物理統計
        stats_text = (
            f"Active: {active_count:5d} ({active_ratio:.1f}%)\n"
            f"MaxAmp: {max_amp:5.1f}\n"
            f"AvgFat: {avg_fat:5.1f}"
        )
        # 疲労度が高いときは赤く警告表示するなどの演出も可能
        stats_color = 'yellow' if avg_fat < 50 else 'red'
        
        ax.text2D(0.80, 0.95, stats_text, transform=ax.transAxes, 
                  color=stats_color, fontsize=14, family='monospace',
                  bbox=dict(facecolor='black', alpha=0.7, edgecolor=stats_color))

        # レイアウト
        ax.set_title(f"R-STN C++ Engine: {name}", color='white', pad=20)
        ax.set_xlim(-1, size); ax.set_ylim(-1, size); ax.set_zlim(-1, size)
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.grid(False)
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.view_init(elev=25, azim=45)
        
        # 保存
        out_name = f"{name}_step{real_step:03d}.png"
        plt.savefig(out_name, facecolor='black', bbox_inches='tight')
        print(f"  > Generated: {out_name} (Fatigue: {avg_fat:.1f})")

    # 全フレーム出力 (動画にするならここをループ)
    # 今回は確認用に 数枚ピックアップ
    target_indices = [0, 5, 20, frames-1] # indexベース (x interval = real step)
    for i in target_indices:
        if i < frames:
            save_frame(i)
        
    plt.close()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(BASE_DIR, "..", "data", "cpp_output")
    if os.path.exists(target_dir):
        for f in sorted(os.listdir(target_dir)):
            if f.endswith(".npz"):
                visualize(os.path.join(target_dir, f))