import sys
import os
import numpy as np
import matplotlib
# GUI環境がないサーバー/仮想環境でも動作させるための設定
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# rstnモジュールを正しくインポートするための設定
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import rstn_cpp

def run_ultimate_lifecycle_v3():
    # --- 1. 物理パラメータの設定 ---
    # 質量 0.99, 粘性 0.35 (ゲル状) のノードを生成 (Box size=1 で代用)
    box = rstn_cpp.RSTNBox(1, seed=123)
    
    # パラメータ設定 (C++版のパラメータオブジェクト経由)
    box.params.inertia = 0.99
    box.params.viscosity = 0.35
    box.params.update_derived() # 必須
    
    # 疲労限界の参照用 (デフォルト値を取得)
    # C++版では fatigue_lim_min/max の範囲で決まるが、ここでは目安として平均値を使う
    fatigue_limit_ref = (box.params.fatigue_lim_min + box.params.fatigue_lim_max) / 2.0

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
        
        # 入力の作成
        inputs = []
        if is_active:
            # 強度100で外部信号を注入 (Index 0)
            inputs.append((0, (100.0, noisy_target)))
        
        # 物理演算実行
        prev_fatigue = box.get_fatigue()[0] if frame > 0 else 0.0
        
        box.step(inputs, is_learning=True)
        
        # 状態取得
        current_f = box.get_frequencies()[0]
        current_fatigue = box.get_fatigue()[0]
        current_amp = box.get_amplitudes()[0]
        
        # 履歴の保存
        f_history.append(current_f)
        # 信号停止中はターゲットを描画しない（nan）
        target_history.append(base_target if is_active else np.nan)
        fatigue_history.append(current_fatigue)
        amplitude_history.append(current_amp)
        
        # 簡易転生判定: 疲労度が閾値付近から急激に落ちた場合
        rebirth_occurred = (prev_fatigue > fatigue_limit_ref * 0.8) and (current_fatigue < prev_fatigue * 0.5)

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
        ax2.axhline(fatigue_limit_ref, color='red', linestyle=':', alpha=0.6, label='Death Limit')
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