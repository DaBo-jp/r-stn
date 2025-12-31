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
    accum_t = 0.0
    
    for s in range(400):
        inputs = []
        for z in range(size):
            for y in range(size):
                idx1 = 0 + y*size + z*(size**2)
                idx2 = (size-1) + y*size + z*(size**2)
                inputs.append((idx1, (100.0, -30.0)))
                inputs.append((idx2, (100.0, 30.0)))
        
        t0 = time.perf_counter()
        box.step(inputs, is_learning=True)
        t1 = time.perf_counter()
        
        accum_t += (t1 - t0)
        history_f.append(box.get_frequencies().copy())
        history_a.append(box.get_amplitudes().copy())
        compute_times.append(accum_t)

    np.savez("case2_data.npz", 
             freqs=np.array(history_f, dtype=np.float32), 
             amps=np.array(history_a, dtype=np.float32), 
             compute_times=np.array(compute_times), 
             name="Case2_Territory", 
             size=size)
    print(f"Case2 Finished. Compute Time: {accum_t:.6f}s")

if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 32)