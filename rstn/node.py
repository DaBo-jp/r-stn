import numpy as np

class RSTNNode:
    """
    R-STN 基礎理論 (Ver 1.10)
    [多様性維持シード / 物理シーケンス連鎖型 resonance インターフェース]
    """
    def __init__(self, f_init=None, seed=None, node_id=0, inertia=0.99, viscosity=0.35):
        # 4.3. 座標依存シードによる多様性の確保
        actual_seed = (seed + node_id) if seed is not None else None
        self.rng = np.random.default_rng(actual_seed)
        
        # 3.1. ノード状態
        self.f_self = f_init if f_init is not None else self.rng.uniform(-50.0, 50.0)
        self.amplitude = 0.0
        self.v_f = 0.0
        
        # 3.3. 物理・代謝パラメータ
        self.inertia = inertia
        self.viscosity = viscosity
        self.dead_band = 1.0 
        self.fatigue = 0.0
        self.fatigue_limit = self.rng.uniform(900.0, 1100.0)
        self.c_load = 10.0          
        self.c_recover = 15.0       
        self.a_threshold = 1.0      
        self.stagnation_count = 0   
        
        # 4.1. 演算定数
        self.sigma_ex = 15.0        
        self.sigma_learn = 15.0     
        self.a_limit = 100.0        

    def resonance(self, a_syn, f_syn, is_learning=True):
        """ 論文 Ver 1.10 準拠の物理シーケンス実行 """
        if is_learning:
            # 学習時: 適応(RFA) -> 励起(透過) -> 代謝(評価)
            force = self.rfa_update(f_syn, a_syn)
            self.gaussian_excitation(a_syn, f_syn)
            return self.metabolic_turnover(force), force
        else:
            # 推論時: 励起(透過) -> 代謝(評価)
            self.gaussian_excitation(a_syn, f_syn)
            return self.metabolic_turnover(0.0), 0.0

    def pull_and_synthesize(self, neighbor_signals):
        """ 4.1. 場の能動的合成 """
        if not neighbor_signals: return 0.0, self.f_self
        s_total = sum(s[0] for s in neighbor_signals)
        sum_abs_a = sum(abs(s[0]) for s in neighbor_signals) + 1e-9
        f_syn = sum(abs(s[0]) * s[1] for s in neighbor_signals) / sum_abs_a
        return abs(s_total), f_syn

    def gaussian_excitation(self, a_syn, f_syn):
        """ 3.2. 共鳴透過型励起 """
        efficiency = np.exp(-((f_syn - self.f_self)**2) / (2 * self.sigma_ex**2))
        self.amplitude = min(a_syn * efficiency, self.a_limit)

    def rfa_update(self, f_syn, a_syn):
        """ 3.3. 慣性・粘性・不感帯を伴うRFA """
        delta_f = f_syn - self.f_self
        if abs(delta_f) < self.dead_band:
            force = 0.0
        else:
            force = np.sign(delta_f) * (a_syn * np.exp(-(delta_f**2) / (2 * self.sigma_learn**2)))
        
        self.v_f = (self.v_f * self.inertia) + (force * (1.0 - self.inertia))
        self.v_f *= self.viscosity
        self.f_self += self.v_f
        return force

    def metabolic_turnover(self, force):
        """ 3.3. 活動依存性代謝回転 """
        # 安定発振時に疲労、変化中・休息中に回復
        if abs(force) < self.dead_band:
            self.fatigue += self.c_load * (self.amplitude / self.a_limit)
        else:
            self.fatigue = max(0.0, self.fatigue - self.c_recover)

        if self.amplitude < self.a_threshold:
            self.fatigue = max(0.0, self.fatigue - self.c_recover)
            self.stagnation_count += (1 if abs(force) < 1e-6 else 0)
        else:
            self.stagnation_count = 0
            
        if self.fatigue > self.fatigue_limit or self.stagnation_count > 150:
            self.f_self = self.rng.uniform(-50.0, 50.0)
            self.fatigue = self.v_f = self.amplitude = self.stagnation_count = 0.0
            return True
        return False