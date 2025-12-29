import sys
import numpy as np
import time
from rstn.box import RSTNBox

# =========================================================================

def run_experiment(size=4, steps=500, target_f=20.0, output_name="exp_data"):
    box = RSTNBox(size=size)
    history = []
    print(f"Running Experiment: Size={size}, Target={target_f}Hz, Steps={steps}")

    for s in range(steps):
        # 入力面 (Z=0) の N*N 個のノードに信号を注入
        input_count = size * size
        inputs = {i: (100.0, target_f) for i in range(input_count)}
        history.append(box.step(inputs, is_learning=True))

    np.savez(f"{output_name}.npz", data=np.array(history), name=output_name, size=size)
    print(f"Saved to {output_name}.npz")

if __name__ == "__main__":
    # 使用例: python3 sim_runner.py 4 500 20.0 case1_4x4
    sz = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run_experiment(size=sz, output_name=f"case1_{sz}x{sz}")