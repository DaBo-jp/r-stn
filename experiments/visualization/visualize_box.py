import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(file_path):
    if not os.path.exists(file_path): return
    loader = np.load(file_path)
    history = loader['data']
    compute_times = loader.get('compute_times', np.zeros(len(history)))
    name, size = str(loader['name']), int(loader['size'])
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    coords = np.array([[z, y, x] for z in range(size) for y in range(size) for x in range(size)])

    def draw_frame(t, save_path=None):
        ax.clear()
        step_data = history[t]
        
        # 散布図プロット
        sc = ax.scatter(coords[:, 2], coords[:, 1], coords[:, 0], 
                       c=step_data[:, 0], s=step_data[:, 1]*2 + 5, 
                       cmap='coolwarm', vmin=-50, vmax=50, 
                       edgecolors='none', alpha=0.7)

        # ラベルと時間表示
        ax.text2D(0.02, 0.95, f"Compute Time: {compute_times[t]:.6f} s", 
                  transform=ax.transAxes, color='yellow', fontsize=14, 
                  weight='bold', bbox=dict(facecolor='black', alpha=0.7))
        
        ax.set_title(f"R-STN {name} (N={size})\nStep: {t}", color='white', pad=20)
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.view_init(elev=20, azim=45)
        
        if save_path: plt.savefig(save_path, facecolor='black', bbox_inches='tight')

    print(f"  [PNG] Generating stills for {name}...")
    # 最初、200ステップ目、最後を抽出
    for i in [0, 200, len(history)-1]:
        draw_frame(i, save_path=f"{name}_step{i}.png")
    plt.close()

if __name__ == "__main__":
    visualize(sys.argv[1])