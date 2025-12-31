import sys
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# rstnモジュールを正しくインポートするための設定
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import rstn_cpp

class RSTNDualReporter:
    def __init__(self):
        self.phases = [
            ("1", "Initial_Tracking"),
            ("2", "Q-Curve_Sweep"),
            ("3", "Sine_Wave_Tracking"),
            ("4", "Noise_Robustness"),
            ("5", "Overwork_Rebirth"),
            ("6", "Recovery_Stagnation")
        ]
        if not os.path.exists("reports"):
            os.makedirs("reports")

    def get_signal_logic(self, phase_id, step, mode):
        """ 
        mode='cont' なら常時ON、mode='int' なら断続ON/OFF 
        """
        is_active = True if mode == 'cont' else ((step // 30) % 2 == 1)
        
        base_f, noisy_f = 0.0, 0.0
        if phase_id == "1": base_f = -10.0
        elif phase_id == "2": base_f = (step / 10.0) - 50.0 
        elif phase_id == "3": base_f = 40.0 * np.sin(2 * np.pi * step / 500.0)
        elif phase_id == "4": 
            base_f = 25.0
            noisy_f = base_f + np.random.uniform(-25, 25) if 400 < step < 800 else base_f
        elif phase_id == "5": base_f = 30.0
        elif phase_id == "6": base_f = 20.0
        
        if phase_id != "4": noisy_f = base_f
        return base_f, noisy_f, is_active

    def run_all(self):
        print("=== R-STN Dual Mode (Continuous & Intermittent) Reporting Starting ===")
        
        # マスターシードの設定
        master_seed = 42

        for p_id, p_name in self.phases:
            for mode in ['cont', 'int']:
                mode_str = "Continuous" if mode == 'cont' else "Intermittent"
                print(f"\n[Processing Phase {p_id}: {p_name} ({mode_str})]")
                
                # node_id=0 で個体初期化 (Box size=1)
                box = rstn_cpp.RSTNBox(1, seed=master_seed)
                box.params.inertia = 0.99
                box.params.viscosity = 0.35
                box.params.dead_band = 1.0 
                box.params.update_derived()

                # 疲労限界リファレンス
                fatigue_limit_ref = (box.params.fatigue_lim_min + box.params.fatigue_lim_max) / 2.0
                
                steps = 1200
                data = {"f": [], "t": [], "fat": [], "rebirth": []}

                for s in range(steps):
                    base_f, noisy_f, active = self.get_signal_logic(p_id, s, mode)
                    
                    # 入力リスト作成
                    inputs = []
                    if active:
                        inputs.append((0, (100.0, noisy_f)))
                    
                    prev_fat = box.get_fatigue()[0] if s > 0 else 0.0
                    
                    # ステップ実行
                    box.step(inputs, is_learning=True)
                    
                    curr_f = box.get_frequencies()[0]
                    curr_fat = box.get_fatigue()[0]
                    
                    # 簡易転生検知
                    if (prev_fat > fatigue_limit_ref * 0.8) and (curr_fat < prev_fat * 0.5):
                        data["rebirth"].append(s)

                    data["f"].append(curr_f)
                    data["t"].append(base_f if active else np.nan)
                    data["fat"].append(curr_fat)

                # --- 保存ファイル名の設定 ---
                base_name = f"reports/phase{p_id}_{mode}"
                
                # --- PNG保存 ---
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                ax1.plot(data["f"], color='green', label='Node Freq', lw=1.5)
                ax1.plot(data["t"], 'r--', alpha=0.3, label='Target')
                ax1.set_title(f"Phase {p_id}: {p_name} ({mode_str})")
                ax1.set_ylim(-65, 65)
                ax1.legend(loc='upper right')
                
                ax2.plot(data["fat"], color='purple', label='Fatigue')
                ax2.axhline(fatigue_limit_ref, color='red', ls=':', alpha=0.5, label='Death Limit')
                ax2.set_title("Metabolic Fatigue")
                ax2.set_ylim(0, 1300)
                ax2.legend(loc='upper right')
                
                for rb in data["rebirth"]:
                    ax1.axvline(rb, color='orange', alpha=0.4, ls='--')
                    ax2.axvline(rb, color='orange', alpha=0.4, ls='--')
                
                plt.tight_layout()
                plt.savefig(f"{base_name}.png")
                plt.close()

                # --- GIF保存 (10ステップ飛ばし) ---
                fig_anim, ax_anim = plt.subplots(figsize=(8, 4))
                line_f, = ax_anim.plot([], [], color='green', lw=1.5)
                line_t, = ax_anim.plot([], [], 'r--', alpha=0.3)
                ax_anim.set_xlim(0, 1200); ax_anim.set_ylim(-65, 65)
                ax_anim.set_title(f"Anim: {p_name} ({mode_str})")

                def anim_func(i):
                    idx = i * 10
                    if idx >= len(data["f"]): idx = len(data["f"]) - 1
                    line_f.set_data(range(idx), data["f"][:idx])
                    line_t.set_data(range(idx), data["t"][:idx])
                    return line_f, line_t

                ani = FuncAnimation(fig_anim, anim_func, frames=len(data["f"])//10, interval=20, blit=True)
                ani.save(f"{base_name}.gif", writer='pillow', fps=30)
                plt.close()
                print(f"  [Saved] {base_name}.png/gif")

        print("\n=== All Dual Mode Reports Generated Successfully ===")

if __name__ == "__main__":
    reporter = RSTNDualReporter()
    reporter.run_all()