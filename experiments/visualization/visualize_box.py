import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(file_path):
    if not os.path.exists(file_path): return
    
    # reportsディレクトリの準備
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    loader = np.load(file_path)
    
    # データ読み込み分岐（新旧対応）
    if 'freqs' in loader and 'amps' in loader:
        # 新形式: (Steps, N*N*N)
        freqs = loader['freqs']
        amps = loader['amps']
        # 以前のフォーマットに合わせて結合 (Steps, Nodes, 2) -> 0:Freq, 1:Amp
        # ただし可視化では FreqとAmpが別々にあればよいので、結合せず使う
    elif 'data' in loader:
        # 旧形式: (Steps, Nodes) の辞書配列だったが、np.savezで保存された時点で構造化配列か、あるいは (Steps, Nodes, 2) になっている可能性がある
        # ここでは visualize_box.py の元のコードが step_data[:, 0] (Freq), step_data[:, 1] (Amp) としているので
        # それに合わせてデータを整形する
        raw_data = loader['data']
        # もし辞書オブジェクトの配列として保存されている場合（旧仕様）、復元が面倒だが
        # 今回の修正で全て (Steps, Nodes, 2) のような形、あるいは freqs/amps 分離形式に変えたので
        # ここでは「新形式」を優先し、旧形式は一旦エラーにするか簡易対応にとどめる
        # ※以前のコードは step_data が (Nodes, 2) であることを期待していた
        print("Warning: Old data format detected. Visualization might fail.")
        freqs = raw_data[:, :, 0]
        amps = raw_data[:, :, 1]
    else:
        print(f"Error: Unknown data format in {file_path}")
        return

    compute_times = loader.get('compute_times', np.zeros(len(freqs)))
    name, size = str(loader['name']), int(loader['size'])
    
    # 3D座標の事前計算
    # Z, Y, X の順序でフラット化されている前提
    coords = np.array([[z, y, x] for z in range(size) for y in range(size) for x in range(size)])

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    def draw_frame(t, save_path=None):
        ax.clear()
        
        # ステップ t のデータ
        step_freqs = freqs[t]
        step_amps = amps[t]
        
        # 散布図プロット
        # X, Y, Z を入れ替えて見やすくする工夫が必要かもしれないが、一旦元のまま
        # coords: (Nodes, 3) -> col 2=X, col 1=Y, col 0=Z
        sc = ax.scatter(coords[:, 2], coords[:, 1], coords[:, 0], 
                       c=step_freqs,           # 色: 周波数
                       s=step_amps * 2 + 5,    # サイズ: 振幅
                       cmap='coolwarm', vmin=-50, vmax=50, 
                       edgecolors='none', alpha=0.7)

        # ラベルと時間表示
        if t < len(compute_times):
            time_val = compute_times[t]
        else:
            time_val = 0.0
            
        ax.text2D(0.02, 0.95, f"Compute Time: {time_val:.6f} s", 
                  transform=ax.transAxes, color='yellow', fontsize=14, 
                  weight='bold', bbox=dict(facecolor='black', alpha=0.7))
        
        ax.set_title(f"R-STN {name} (N={size})\nStep: {t}", color='white', pad=20)
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        
        # 視点固定
        ax.view_init(elev=20, azim=45)
        
        # 軸ラベル等は黒背景だと見えないので白にするか、axis offにする
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        
        if save_path: 
            plt.savefig(save_path, facecolor='black', bbox_inches='tight')
            print(f"    Saved: {os.path.basename(save_path)}")

    print(f"  [PNG] Generating stills for {name} to reports/...")
    
    # 最初、中間、最後を抽出
    steps_to_save = [0, len(freqs)//2, len(freqs)-1]
    
    for i in steps_to_save:
        if i < len(freqs):
            save_name = f"{name}_step{i}.png"
            full_path = os.path.join(reports_dir, save_name)
            draw_frame(i, save_path=full_path)
            
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize(sys.argv[1])
    else:
        print("Usage: python3 visualize_box.py <path_to_npz>")