import sys
import time
import numpy as np
from rstn.box import RSTNBox

def run(size=32):
    box = RSTNBox(size=size)
    history, compute_times = [], []
    accum_t = 0.0
    
    for s in range(400):
        inputs = {}
        for z in range(size):
            for y in range(size):
                inputs[0 + y*size + z*(size**2)] = (100.0, -30.0)
                inputs[(size-1) + y*size + z*(size**2)] = (100.0, 30.0)
        
        t0 = time.perf_counter()
        res = box.step(inputs, is_learning=True)
        t1 = time.perf_counter()
        
        accum_t += (t1 - t0)
        history.append(res)
        compute_times.append(accum_t)

    np.savez("case2_data.npz", data=np.array(history), compute_times=np.array(compute_times), name="Case2_Territory", size=size)
    print(f"Case2 Finished. Compute Time: {accum_t:.6f}s")

if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 32)