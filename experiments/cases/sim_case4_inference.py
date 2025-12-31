import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import time
import numpy as np
import rstn_cpp

def run(size=32):
    box = rstn_cpp.RSTNBox(size, seed=42)
    history_f, history_a = [], []
    compute_times = []
    input_indices = range(size * size)
    accum_t = 0.0
    
    for s in range(400):
        is_learn = (s < 200)
        freq = 20.0 if is_learn else -40.0
        inputs = [(i, (100.0, freq)) for i in input_indices]
        
        t0 = time.perf_counter()
        box.step(inputs, is_learning=is_learn)
        t1 = time.perf_counter()
        
        accum_t += (t1 - t0)
        history_f.append(box.get_frequencies().copy())
        history_a.append(box.get_amplitudes().copy())
        compute_times.append(accum_t)

    np.savez("case4_data.npz", 
             freqs=np.array(history_f, dtype=np.float32), 
             amps=np.array(history_a, dtype=np.float32), 
             compute_times=np.array(compute_times), 
             name="Case4_Inference", 
             size=size)
    print(f"Case4 Finished. Compute Time: {accum_t:.6f}s")

if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 32)