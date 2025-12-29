import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def visualize_dynamic(file_path):
    if not os.path.exists(file_path): return
    print(f"Visualizing {file_path} ...")
    
    loader = np.load(file_path)
    freqs = loader['freqs'] 
    amps = loader['amps']
    size = int(loader['size'])
    interval = int(loader['save_interval'])
    
    frames = freqs.shape[0]
    
    # 座標
    coords = np.zeros((size**3, 3))
    for z in range(size):
        for y in range(size):
            for x in range(size):
                idx = x + y*size + z*size*size
                coords[idx] = [z, y, x]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 出力先
    out_dir = "analysis_Case5"
    os.makedirs(out_dir, exist_ok=True)

    for i in range(frames):
        ax.clear()
        step = i * interval
        
        # ターゲット位置の計算 (描画用)
        cy, cx = size // 2, size // 2
        radius = size * 0.35
        angle = (step / 100.0) * 2 * math.pi
        ty = cy + int(radius * math.sin(angle))
        tx = cx + int(radius * math.cos(angle))
        
        # パス描画 (20Hz付近のみ抽出)
        # 振幅閾値を高めにして「主要パス」だけ浮かび上がらせる
        mask = (amps[i] > 10.0) & (np.abs(freqs[i] - 20.0) < 10.0)
        
        if np.any(mask):
            c = coords[mask]
            # Z軸(奥行)を横軸(X)にして、左(Src)→右(Tgt)の流れで見せる
            ax.scatter(c[:,0], c[:,1], c[:,2], 
                       c=freqs[i][mask], cmap='coolwarm', vmin=-50, vmax=50, 
                       s=amps[i][mask]*0.5, alpha=0.6)
            
        # マーカー: Start(緑) & Goal(赤)
        ax.scatter([0], [cy], [cx], c='lime', s=200, marker='*', label='Source')
        ax.scatter([size-1], [ty], [tx], c='magenta', s=200, marker='X', label='Target')
        
        ax.set_xlim(0, size); ax.set_ylim(0, size); ax.set_zlim(0, size)
        ax.set_title(f"Dynamic Re-routing | Step: {step}")
        ax.set_xlabel("Z (Depth)"); ax.set_ylabel("Y"); ax.set_zlabel("X")
        ax.set_facecolor('black')
        ax.grid(False)
        fig.patch.set_facecolor('black')
        
        # 視点を横からに固定
        ax.view_init(elev=10, azim=-100)
        
        out_name = os.path.join(out_dir, f"frame_{step:04d}.png")
        plt.savefig(out_name, facecolor='black')
        if i % 10 == 0: print(f"Saved {out_name}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "..", "data", "cpp_output", "Case5_Dynamic.npz")
    visualize_dynamic(data_path)