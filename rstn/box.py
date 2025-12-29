import numpy as np
from .node import RSTNNode

class RSTNBox:
    def __init__(self, size=32, seed=42):
        self.size = size
        self.nodes = []
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    node_id = x + y * size + z * (size**2)
                    f_init = np.random.uniform(-40, 40)
                    node = RSTNNode(f_init=f_init, seed=seed, node_id=node_id)
                    # 大規模系向けの標準パラメータ
                    node.sigma_ex = 20.0
                    node.sigma_learn = 20.0
                    node.inertia = 0.95
                    node.viscosity = 0.5
                    self.nodes.append(node)

    def get_neighbors(self, idx):
        size = self.size
        z, y, x = idx // (size**2), (idx // size) % size, idx % size
        neighbors = []
        # 6近傍（前後左右上下）参照
        for dz, dy, dx in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nz, ny, nx = z+dz, y+dy, x+dx
            if 0 <= nz < size and 0 <= ny < size and 0 <= nx < size:
                neighbors.append(nz * (size**2) + ny * size + nx)
        return neighbors

    def step(self, inputs, is_learning=True):
        current_states = [(n.amplitude, n.f_self) for n in self.nodes]
        step_results = []
        for i, node in enumerate(self.nodes):
            if i in inputs:
                a_syn, f_syn = inputs[i]
            else:
                neighbor_data = [current_states[nb] for nb in self.get_neighbors(i)]
                a_syn, f_syn = node.pull_and_synthesize(neighbor_data)
            
            # 物理シーケンス実行
            rebirth, force = node.resonance(a_syn, f_syn, is_learning=is_learning)
            step_results.append([node.f_self, node.amplitude, node.fatigue, float(rebirth)])
        return np.array(step_results)