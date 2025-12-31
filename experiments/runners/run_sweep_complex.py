import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import rstn_cpp
import numpy as np
import time
import math
import itertools
import concurrent.futures

# =========================================================================
# 設定 & パラメータ範囲
# =========================================================================
N = 32
STEPS = 200
DATA_DIR = "experiment_data"

# ★保存判定の閾値 (Sticky Path基準)
# 100点満点からの減点方式。
# 経路が途切れると一気に30〜50点引かれるため、75点以上なら
# 「遅れはあるかもしれないが、確実に線がつながっている」モデルと言える。
SAVE_THRESHOLD = 75.0 

# ベースラインパラメータ
BASE_PARAMS = {
    'attenuation': 0.10,
    'sigma_ex': 10.0,
    'sigma_learn': 20.0,
    'inertia': 0.90,
    'viscosity': 0.30,
    'dead_band': 1.0,
    'c_load': 10.0,
    'c_recover': 15.0,
    'a_threshold': 1.0,
    'a_limit': 100.0
}

# スイープ条件 (微小値epsを追加して終点を含める)
eps = 1e-9
RANGE_VISCOSITY   = np.arange(0.5, 0.8 + eps, 0.05) # 最重要：内側ループ
RANGE_INERTIA     = np.arange(0.5, 1.0 + eps, 0.05)
RANGE_ATTENUATION = np.arange(0.0, 1.0 + eps, 0.2)  # 寄与度低：外側ループ
RANGE_RESONANCE   = np.arange(1, 30 + eps, 1)

# =========================================================================
# ロジック関数群
# =========================================================================

def apply_params(box, params_dict):
    p = box.params
    for k, v in params_dict.items():
        if hasattr(p, k):
            setattr(p, k, v)
    p.update_derived()

def get_target_pos(s, size):
    """ターゲットの円運動軌跡"""
    cy, cx = size // 2, size // 2
    radius = size * 0.35
    angle = (s / float(STEPS)) * 2 * math.pi 
    
    ty = cy + int(radius * math.sin(angle))
    tx = cx + int(radius * math.cos(angle))
    
    ty = max(0, min(size-1, ty))
    tx = max(0, min(size-1, tx))
    return tx, ty

def inputs_case5(s, size):
    inputs = []
    # 1. Source (Center)
    cy, cx = size // 2, size // 2
    idx_src = cx + cy * size + 0 * (size**2)
    inputs.append((idx_src, (100.0, 20.0)))
    
    # 2. Target (Moving)
    tx, ty = get_target_pos(s, size)
    idx_tgt = tx + ty * size + (size-1) * (size**2)
    inputs.append((idx_tgt, (100.0, 20.0)))
    return inputs

def get_filename(v, i, a, r):
    return f"Visc{v:.2f}_Inert{i:.2f}_Attn{a:.2f}_Res{r:02d}.npz"

def quick_evaluate(amps_np, size):
    """
    【評価関数: Sticky Path Metrics】
    推論フェーズでの再現性を重視し、「経路の連続性」と「追従性」を評価する。
    
    戻り値: 0.0 〜 100.0 のスコア
    """
    # パラメータ設定
    JUMP_PENALTY = 40.0   # ワープ（経路断絶）に対する重いペナルティ
    LAG_TOLERANCE = 8.0   # 遅れ許容範囲 (マス)
    LAG_PENALTY_RATE = 2.0 # 許容範囲を超えた距離1マスあたりの減点
    SIGNAL_DEATH_PENALTY = 10.0 # 信号消失時の減点
    
    score_deduction = 0.0 # 減点累積
    valid_steps = 0
    
    # 状態記憶用 (前のピーク位置)
    prev_px, prev_py = -1, -1
    
    # 後半50ステップ(定常状態)で評価
    check_range = range(STEPS - 50, STEPS)

    # --- 3Dデータの整形 (Max Projection) ---
    total_elements = amps_np.size
    elements_per_step = total_elements // STEPS
    
    try:
        if elements_per_step == size * size:
            grids = amps_np.reshape(STEPS, size, size)
        elif elements_per_step == size * size * size:
            # (Steps, Depth, H, W) -> (Steps, H, W)
            raw_3d = amps_np.reshape(STEPS, size, size, size)
            grids = np.max(raw_3d, axis=1) 
        else:
            return 0.0 
    except:
        return 0.0

    # --- 評価ループ ---
    for s in check_range:
        tx, ty = get_target_pos(s, size)
        grid = grids[s]
        
        # 1. 信号生存チェック
        max_val = np.max(grid)
        if max_val < 0.1: 
            score_deduction += SIGNAL_DEATH_PENALTY
            prev_px, prev_py = -1, -1 # パス切れ扱い
            valid_steps += 1
            continue

        # ピーク位置特定
        peak_idx = np.argmax(grid)
        py, px = np.unravel_index(peak_idx, (size, size))
        
        # 2. 連続性チェック (Connectivity)
        if prev_px != -1:
            jump_dist = math.sqrt((px - prev_px)**2 + (py - prev_py)**2)
            # 2.5マス以上のジャンプは「経路断絶」とみなす
            if jump_dist > 2.5:
                score_deduction += JUMP_PENALTY
        
        # 状態更新
        prev_px, prev_py = px, py

        # 3. 追従性チェック (Lag Tolerance)
        dist_tgt = math.sqrt((px - tx)**2 + (py - ty)**2)
        
        # 許容範囲を超えた分だけ減点
        if dist_tgt > LAG_TOLERANCE:
            excess = dist_tgt - LAG_TOLERANCE
            score_deduction += excess * LAG_PENALTY_RATE

        valid_steps += 1
    
    if valid_steps == 0: return 0.0
    
    # 平均減点を算出
    avg_deduction = score_deduction / len(check_range)
    final_score = max(0.0, 100.0 - avg_deduction)
    
    return final_score

def save_worker(filepath, payload):
    try:
        np.savez_compressed(filepath, **payload)
        return True
    except Exception as e:
        print(f"[Error] Save failed for {filepath}: {e}")
        return False

# =========================================================================
# Phase 1: シミュレーション実行 (フィルタリング & 非同期保存)
# =========================================================================
def run_simulation_phase():
    print("=== Phase 1: Simulation Batch (Sticky Path Evaluation) ===")
    print(f"Save Threshold: Score >= {SAVE_THRESHOLD}")
    print("Loop Order: Attn(Out) -> Res -> Inert -> Visc(In)")
    os.makedirs(DATA_DIR, exist_ok=True)

    # ループ順序: 外側(遅い) -> 内側(速い)
    # Attenuation(環境) -> Resonance(入力) -> Inertia(反応) -> Viscosity(核心)
    combinations = list(itertools.product(
        RANGE_ATTENUATION, 
        RANGE_RESONANCE,    
        RANGE_INERTIA,      
        RANGE_VISCOSITY     
    ))
    total = len(combinations)
    print(f"Total Combinations: {total}")

    stats = {"saved": 0, "rejected": 0}

    # 非同期保存用Executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        start_time = time.time()
        print("Starting simulation... (Press Ctrl+C to abort)")

        for idx, (a, r, i, v) in enumerate(combinations):
            filename = get_filename(v, i, a, int(r))
            filepath = os.path.join(DATA_DIR, filename)

            # 再開機能
            if os.path.exists(filepath):
                if idx % 500 == 0: print(f"[{idx+1}/{total}] Skip (Exists): {filename}")
                continue

            # --- シミュレーション ---
            current_params = BASE_PARAMS.copy()
            current_params['viscosity'] = v
            current_params['inertia'] = i
            current_params['attenuation'] = a
            current_params['sigma_ex'] = r 

            box = rstn_cpp.RSTNBox(N, seed=42)
            apply_params(box, current_params)
            
            history_a = []
            for s in range(STEPS):
                inputs = inputs_case5(s, N)
                box.step(inputs, is_learning=True)
                history_a.append(box.get_amplitudes().copy())

            # --- 評価 & 足切り ---
            amps_np = np.array(history_a, dtype=np.float16)
            
            # グリア脳基準で評価
            score = quick_evaluate(amps_np, N)
            
            # ログ用コンテキスト文字列
            param_str = f"Attn={a:.2f} Res={int(r):02d} Inert={i:.2f} Visc={v:.2f}"
            
            if score < SAVE_THRESHOLD:
                stats["rejected"] += 1
                # 不合格ログ (間引いて表示)
                if (idx + 1) % 200 == 0:
                     print(f"[{idx+1}/{total}] Rejected: {param_str} (Score {score:.1f})")
                continue

            # --- 合格：非同期保存 ---
            stats["saved"] += 1
            payload = {
                'amps': amps_np,
                'params': str(current_params),
                'visc': v, 'inert': i, 'attn': a, 'res': r,
                'score': score
            }

            future = executor.submit(save_worker, filepath, payload)
            futures.append(future)

            # メモリ掃除
            if len(futures) > 100:
                futures = [f for f in futures if not f.done()]

            # --- 進捗ログ表示 ---
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                avg_t = elapsed / (idx + 1)
                eta = avg_t * (total - idx)
                rate = stats["saved"] / (stats["saved"] + stats["rejected"] + 1e-9) * 100
                
                print(f"[{idx+1}/{total}] {param_str} | Saved: {stats['saved']} ({rate:.1f}%) | ETA: {eta/60:.1f} min")

        print("Waiting for pending saves...")
    
    print("\nPhase 1 Complete.")
    print(f"Final Stats: Saved={stats['saved']}, Rejected={stats['rejected']}")

# =========================================================================
# メイン実行ブロック
# =========================================================================
if __name__ == "__main__":
    run_simulation_phase()
    
    print("\n" + "="*50)
    print(" Simulation Finished.")
    print(" Promising 'High-Viscosity' candidates have been saved.")
    print("="*50)