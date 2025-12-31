import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import time
import rstn_cpp

# =========================================================================

def run_experiment(size=4, steps=500, target_f=20.0, output_name="exp_data"):
    box = rstn_cpp.RSTNBox(size, seed=42)
    history_f, history_a = [], []
    print(f"Running Experiment: Size={size}, Target={target_f}Hz, Steps={steps}")

    input_count = size * size
    
    for s in range(steps):
        # 入力面 (Z=0) の N*N 個のノードに信号を注入
        inputs = [(i, (100.0, target_f)) for i in range(input_count)]
        box.step(inputs, is_learning=True)
        
        history_f.append(box.get_frequencies().copy())
        history_a.append(box.get_amplitudes().copy())

    np.savez(f"{output_name}.npz", 
             freqs=np.array(history_f, dtype=np.float32), 
             amps=np.array(history_a, dtype=np.float32), 
             name=output_name, 
             size=size)
    print(f"Saved to {output_name}.npz")

if __name__ == "__main__":
    # 使用例: python3 sim_runner.py 4 500 20.0 case1_4x4
    sz = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run_experiment(size=sz, output_name=f"case1_{sz}x{sz}")