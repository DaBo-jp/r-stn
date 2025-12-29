import sys
import os
import numpy as np
import matplotlib
# GUI環境がないサーバー/仮想環境でも動作させるための設定
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 同一ディレクトリにある rstn_node.py を確実に読み込むための設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rstn.node import RSTNNode

def run_ultimate_lifecycle_v3():
    # --- 1. 物理パラメータの設定 ---
    # 質量 0.99, 粘性 0.35 (ゲル状) のノードを生成
    node = RSTNNode(f_init=0.0, seed=123, inertia=0.99, viscosity=0.35)
    
    # 全ライフサイクルをカバーする長尺ステップ
    steps = 2200 
    
    f_history = []
    target_history = []
    fatigue_history = []
    amplitude_history = []
    
    # グラフの初期化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.3)

    def get_signal(step):
        """ 各ステップでの外部信号を定義するシナリオ関数 """
        # Phase 1-4: 非線形往復 (正弦波)
        base_f = 40.0 * np.sin(2 * np.pi * step / 500.0)
        is_active = True
        current_f = base_f
        
        # Phase 5: 強烈なバーストノイズ (Step 800-1100)
        if 800 <= step < 1100:
            current_f += np.random.uniform(-25.0, 25.0)
            
        # Phase 6: 断続的信号による疲労回復検証 (Step 1200-1800)
        if 1200 <= step < 1800:
            # 80ステップ周期で40ステップずつ休息 (Duty 50%)
            if (step // 40) % 2 == 1:
                is_active = False # 休息期間
        
        # 最終沈黙シナリオ: 膠着死を誘発 (Step 2000以降)
        if step >= 2000:
            is_active = False

        return base_f, current_f, is_active

    def update(frame):
        base_target, noisy_target, is_active = get_signal(frame)
        
        # 3.2. プル型発振のシミュレート
        if is_active:
            # 強度100で外部信号を注入
            a_syn, f_syn = node.pull_and_synthesize([(100.0, noisy_target)])
        else:
            # 信号なし（0.0）
            a_syn, f_syn = 0.0, node.f_self

        # 3.2 & 3.3. 物理演算実行
        node.gaussian_excitation(a_syn, f_syn) # 励起
        force = node.rfa_update(f_syn, a_syn)   # 周波数更新 (RFA)
        
        # 履歴の保存
        f_history.append(node.f_self)
        # 信号停止中はターゲットを描画しない（nan）
        target_history.append(base_target if is_active else np.nan)
        fatigue_history.append(node.fatigue)
        amplitude_history.append(node.amplitude)
        
        # 3.3. 代謝回転 (過労死・膠着死)
        rebirth_occurred = node.metabolic_turnover(force)
        
        # 要所でのスナップショット保存
        if frame in [0, 500, 1000, 1500, 2000, 2199]:
            plt.savefig(f"snapshot_step_{frame}.png")
            print(f"Saved snapshot at step {frame}")

        # --- プロットの描画 ---
        # 左：周波数追従（不感帯と慣性の効果を確認）
        ax1.clear()
        ax1.plot(f_history, color='green', lw=1.5, label='Node Frequency')
        ax1.plot(target_history, color='red', linestyle='--', alpha=0.3, label='Target Wave')
        ax1.set_title(f"Step {frame}: Non-linear Tracking")
        ax1.set_ylim(-65, 65)
        ax1.legend(loc='lower right', fontsize='small')
        ax1.grid(True, alpha=0.2)

        # 中：疲労度（断続信号による回復、過労死を確認）
        ax2.clear()
        ax2.plot(fatigue_history, color='purple', lw=1.2)
        ax2.axhline(node.fatigue_limit, color='red', linestyle=':', alpha=0.6, label='Death Limit')
        ax2.set_title("Fatigue & Recovery")
        ax2.set_ylim(0, 1300)
        ax2.grid(True, alpha=0.2)

        # 右：振幅（共鳴状態の確認）
        ax3.clear()
        ax3.plot(amplitude_history, color='blue', lw=1.2)
        ax3.set_title("Resonance Amplitude")
        ax3.set_ylim(0, 110)
        ax3.grid(True, alpha=0.2)

        if rebirth_occurred:
            ax1.annotate('REBIRTH', xy=(len(f_history), f_history[-1]), 
                         color='orange', weight='bold', ha='center', fontsize=10)

    print("Starting Comprehensive Lifecycle Simulation (2200 steps)...")
    ani = FuncAnimation(fig, update, frames=steps, interval=20, repeat=False)
    
    output_filename = "rstn_comprehensive_lifecycle.gif"
    ani.save(output_filename, writer='pillow', fps=40)
    print(f"Simulation Finished. Result saved as {output_filename}")

if __name__ == "__main__":
    run_ultimate_lifecycle_v3()