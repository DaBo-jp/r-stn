import numpy as np
import os
import glob
import math
import pandas as pd

# =========================================================================
# 設定
# =========================================================================
DATA_DIR = "experiment_data"    # NPZがある場所
N = 32
STEPS = 200

# ターゲットの動きを再現する関数（正解データ生成用）
def get_target_pos(s, size):
    # inputs_case5 と同じロジック
    cy, cx = size // 2, size // 2
    radius = size * 0.35
    angle = (s / float(STEPS)) * 2 * math.pi 
    
    ty = cy + int(radius * math.sin(angle))
    tx = cx + int(radius * math.cos(angle))
    
    ty = max(0, min(size-1, ty))
    tx = max(0, min(size-1, tx))
    return tx, ty

# =========================================================================
# 評価ロジック
# =========================================================================
def evaluate_file(filepath):
    try:
        # npz読み込み
        with np.load(filepath, allow_pickle=True) as data:
            amps = data['amps'].astype(np.float32) # (STEPS, ...)
            # パラメータ取得
            v = float(data['visc'])
            i = float(data['inert'])
            a = float(data['attn'])
            r = int(data['res'])

        # 形状チェック & 整形
        # (Steps, H*W) or (Steps, Layers, H*W) -> (Steps, H, W)
        if amps.ndim == 2:
            # (Steps, H*W)
            frames = amps.reshape(STEPS, N, N)
        elif amps.ndim == 3:
            # (Steps, Layers, H*W) -> Max Projectionして (Steps, H, W)
            # RSTNの実装に合わせて調整。ここでは単純化して総和かMaxをとる
            # もし(Steps, N, N)ならそのままでOK
            if amps.shape[1] == N and amps.shape[2] == N:
                frames = amps
            else:
                # 多層の場合の簡易投影
                frames = np.max(amps.reshape(STEPS, -1, N, N), axis=1)
        else:
            return None

        score_intensity = 0.0
        score_focus = 0.0
        score_accuracy = 0.0
        
        valid_steps = 0

        # 初期の立ち上がり(例えば最初の20ステップ)は評価から除外してもよい
        START_EVAL = 20 

        for s in range(START_EVAL, STEPS):
            grid = frames[s]
            tx, ty = get_target_pos(s, N)
            
            # 1. Intensity: ターゲット地点のエネルギー
            val_at_target = grid[ty, tx]
            
            # 2. Focus: 全エネルギーに対するターゲット地点の割合 (SNR的なもの)
            total_energy = np.sum(grid) + 1e-9
            focus_ratio = val_at_target / total_energy
            
            # 3. Accuracy: ピーク位置との距離
            # 最も輝いている点を探す
            py, px = np.unravel_index(np.argmax(grid), grid.shape)
            dist = math.sqrt((px - tx)**2 + (py - ty)**2)
            
            score_intensity += val_at_target
            score_focus += focus_ratio
            score_accuracy += dist # 小さいほど良い
            
            valid_steps += 1

        # 平均化
        avg_intensity = score_intensity / valid_steps
        avg_focus = score_focus / valid_steps
        avg_dist = score_accuracy / valid_steps

        # --- 総合スコア (カスタム) ---
        # 距離が遠い(追従していない)とスコア激減
        # エネルギーが低い(死んでる)とスコア激減
        # Focusが高い(くっきりしている)とプラス
        
        # 距離ペナルティ: 距離が0なら1.0, 距離がNなら0に近づく
        accuracy_factor = max(0, (1.0 - (avg_dist / (N/2))))
        
        # 総合評価関数 (ここを調整して好みの波を探す)
        # Intensity * Focus * Accuracy
        # Focusは値が小さいので重み付けする
        final_score = avg_intensity * (avg_focus * 100) * (accuracy_factor ** 2)

        return {
            "file": os.path.basename(filepath),
            "Visc": v, "Inert": i, "Attn": a, "Res": r,
            "Score": final_score,
            "Avg_Intensity": avg_intensity,
            "Avg_Focus": avg_focus,
            "Avg_Dist": avg_dist
        }

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

# =========================================================================
# メイン実行
# =========================================================================
if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    print(f"Analyzing {len(files)} files...")
    
    results = []
    for idx, f in enumerate(files):
        res = evaluate_file(f)
        if res:
            results.append(res)
        
        if (idx+1) % 1000 == 0:
            print(f"Processed {idx+1} files...")

    if not results:
        print("No results.")
        exit()

    # DataFrame化してソート
    df = pd.DataFrame(results)
    
    # スコアが高い順に並べる
    df_sorted = df.sort_values(by="Score", ascending=False)
    
    # CSV保存
    csv_path = "evaluation_ranking.csv"
    df_sorted.to_csv(csv_path, index=False)
    
    print(f"\nSaved ranking to {csv_path}")
    print("Top 10 Candidates:")
    print(df_sorted[['file', 'Score', 'Avg_Intensity', 'Avg_Dist']].head(10))

    # 推奨: 上位のファイルのパラメータ分布を見る
    print("\nBest Parameter Trends (Top 50 average):")
    print(df_sorted.head(50)[['Visc', 'Inert', 'Attn', 'Res']].mean())