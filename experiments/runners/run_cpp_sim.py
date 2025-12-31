import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import rstn_cpp
import numpy as np
import time
import math

# =========================================================================
# シミュレーション設定
# =========================================================================
N = 32              # 規模 (32推奨)
LOG_INTERVAL = 20   # ログ出力間隔 (コンソール表示用)
SAVE_INTERVAL = 5   # データ保存間隔

# 冷却判定の閾値
# AvgFat (平均疲労度) がこの値を下回るまで「COOL」フェーズを継続する
COOLING_THRESHOLD = 5.0 


def print_header():
    """ コンソール出力用のヘッダー表示 """
    print(f"{'Step':>5} | {'Phase':>10} | {'Input Status':>20} | {'Compute':>9} | {'Active':>7} | {'MaxAmp':>6} | {'AvgFat':>6}")
    print("-" * 95)


def calc_stats_str(box):
    """ C++エンジンから統計情報を取得して整形用の値を返す """
    amps = box.get_amplitudes()
    fats = box.get_fatigue()
    
    max_amp = np.max(amps)
    avg_fat = np.mean(fats)
    active_count = np.count_nonzero(amps > 1.0) 
    
    return active_count, max_amp, avg_fat


def setup_params(box, mode="SuperConductive"):
    """ 
    物理パラメータの一括設定 
    Case 6 用に「超伝導チューニング」を適用する
    """
    if mode == "SuperConductive":
        # --- 超伝導チューニング (Super-Conductive) ---
        # 目的: 粘性を排除し、入力に対して瞬時に反応・整地されるようにする
        
        # 1. 空間減衰率: 0.15
        #    信号が無限に広がらないよう適度に絞る
        box.params.attenuation = 0.15   
        
        # 2. 共鳴帯域: 8.0
        #    鋭いパスを作るため、狭めに設定
        box.params.sigma_ex = 8.0       
        
        # 3. 学習帯域: 20.0
        box.params.sigma_learn = 20.0   
        
        # 4. 慣性: 0.70
        #    軽くする＝反応速度を上げる。ターゲット切り替えに即座に追従させる。
        box.params.inertia = 0.70       
        
        # 5. 粘性: 0.05
        #    抵抗をほぼゼロにする。信号が滑るように伝わる。
        box.params.viscosity = 0.05     
        
        # 6. 不感帯: 1.0
        box.params.dead_band = 1.0
        
        # 7. 代謝パラメータ
        box.params.c_load = 10.0
        box.params.c_recover = 15.0
        box.params.a_threshold = 1.0
        box.params.a_limit = 100.0
    
    # 必須: 派生値の更新 (C++側の係数再計算)
    box.params.update_derived()


def run_case6_discrete():
    """
    Case 6: Discrete Switching 実行関数
    ターゲット移動 -> 残響消滅待ち -> 推論データ取得 -> 次のターゲット
    というサイクルを自動化する。
    """
    case_name = "Case6_Discrete"
    print(f"\n=== Running {case_name} (N={N}) ===")
    
    # 1. Box初期化
    box = rstn_cpp.RSTNBox(N, seed=42)
    
    # 2. パラメータ適用 (超伝導モード)
    setup_params(box, mode="SuperConductive")
    
    # 3. ターゲット座標リスト (中心 -> 右上 -> 左下 -> 中心)
    # [Target Name, (Target Y, Target X)]
    c = N // 2
    targets = [
        ("Center",   (c, c)),
        ("TopRight", (N-4, N-4)),
        ("BtmLeft",  (4, 4)),
        ("Center",   (c, c)) # 最後に戻ってくる
    ]
    
    # ステートマシン変数
    target_idx = 0
    phase = "LEARN"  # 初期フェーズ: LEARN -> COOL -> INFER -> (Next Target)
    
    # 各フェーズの持続カウンタ
    phase_timer = 0
    LEARN_DURATION = 150  # 学習にかける固定ステップ数
    INFER_DURATION = 50   # 推論データを取るステップ数
    
    # データ保存用
    history_f = []
    history_a = []
    compute_times = []
    
    accum_time = 0.0
    
    print_header()

    # 無限ループ防止用の安全リミット
    MAX_TOTAL_STEPS = 2000 
    s = 0
    
    # --- メインループ ---
    while s < MAX_TOTAL_STEPS:
        # 現在のターゲット情報取得
        tgt_name, (ty, tx) = targets[target_idx]
        
        # 入力データの準備
        inputs = []
        is_learning = True
        status_str = ""
        
        # 統計取得 (判断用)
        act, mx, avg_f = calc_stats_str(box)

        # === ステートマシン ===
        if phase == "LEARN":
            # -------------------------------------------------
            # 学習フェーズ: ターゲットに入力 (20Hz)
            # -------------------------------------------------
            is_learning = True
            
            # Source (Center固定)
            idx_src = c + c*N + 0*(N*N) # Z=0
            inputs.append((idx_src, (100.0, 20.0)))
            
            # Target (現在地)
            idx_tgt = tx + ty*N + (N-1)*(N*N) # Z=N-1
            inputs.append((idx_tgt, (100.0, 20.0)))
            
            status_str = f"LEARN -> {tgt_name}"
            
            phase_timer += 1
            if phase_timer >= LEARN_DURATION:
                # 指定時間経過したら冷却へ移行
                phase = "COOL"
                phase_timer = 0
                
        elif phase == "COOL":
            # -------------------------------------------------
            # 冷却フェーズ: 入力なし, 学習モード維持(自然減衰)
            # -------------------------------------------------
            is_learning = True 
            inputs = [] # 入力なし
            
            status_str = f"COOL (Fat:{avg_f:.1f})"
            
            # 終了条件: 疲労度が十分に下がる (残響が消える)
            if avg_f < COOLING_THRESHOLD:
                phase = "INFER"
                phase_timer = 0
                
        elif phase == "INFER":
            # -------------------------------------------------
            # 推論フェーズ: 入力あり(-40Hz), 学習OFF
            # -------------------------------------------------
            is_learning = False
            
            # 推論入力 (Sourceのみに入れてパスの導通を確認)
            idx_src = c + c*N + 0*(N*N)
            inputs.append((idx_src, (100.0, -40.0)))
            
            # 必要ならTarget側にも入れてフィルタ性能を見る場合は以下を追加
            # idx_tgt = tx + ty*N + (N-1)*(N*N)
            # inputs.append((idx_tgt, (100.0, -40.0)))
            
            status_str = f"INFER (-40Hz)"
            
            phase_timer += 1
            if phase_timer >= INFER_DURATION:
                # 推論終了 -> 次のターゲットへ
                target_idx += 1
                if target_idx >= len(targets):
                    print(f"{s:5d} | {'FINISH':>10} | {'All targets done':>20} | {'---':>9} | {'---':>7} | {'---':>6} | {'---':>6}")
                    break # 全ターゲット終了
                
                # 次のターゲットへ向けて学習開始
                phase = "LEARN"
                phase_timer = 0

        # --- 物理演算 (C++ Backend) ---
        t0 = time.perf_counter()
        box.step(inputs, is_learning=is_learning)
        t1 = time.perf_counter()
        
        step_time = t1 - t0
        accum_time += step_time

        # --- ログ出力 ---
        if s % LOG_INTERVAL == 0:
            print(f"{s:5d} | {phase:>10} | {status_str:>20} | {step_time*1000:6.2f} ms | {act:7d} | {mx:6.1f} | {avg_f:6.1f}")

        # --- データ保存 ---
        if s % SAVE_INTERVAL == 0:
            history_f.append(box.get_frequencies().copy())
            history_a.append(box.get_amplitudes().copy())
            compute_times.append(accum_time)
            
        s += 1

    # .npzファイル書き出し
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data", "cpp_output")
    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, f"{case_name}.npz")
    
    np.savez(save_path, 
             freqs=np.array(history_f, dtype=np.float32),
             amps=np.array(history_a, dtype=np.float32),
             compute_times=np.array(compute_times),
             name=case_name,
             size=N,
             save_interval=SAVE_INTERVAL)
    print(f"\n  Saved to {save_path}")


# =========================================================================
# メイン実行部
# =========================================================================
if __name__ == "__main__":
    # 出力先ディレクトリ作成 (experiments/data/cpp_output)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data", "cpp_output")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Case 6 を実行 (保存先パスを調整する必要があるため関数側も少し修正が必要だが、
    # run_case6_discrete 内で data_cpp ハードコードがあるため修正する)
    run_case6_discrete()