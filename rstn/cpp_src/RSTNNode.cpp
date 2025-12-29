#include "RSTNNode.hpp"
#include <cmath>
#include <algorithm>

bool RSTNNode::update_state_lut(
    const RSTNParams& params,
    RSTNState& state,
    const double a_syn,
    const double f_syn,
    const double next_random_f,
    const bool is_learning,
    const double* lut_ex,
    const double* lut_learn,
    const double lut_resolution,
    const int lut_max_idx
) {
    double diff = f_syn - state.f_self;

    // 1. 励起 (Excitation) - LUT
    gaussian_excitation_lut(params, &state.amplitude, diff, a_syn, lut_ex, lut_resolution, lut_max_idx);

    if (is_learning) {
        // 2. 適応 (Adaptation) - LUT
        double force = rfa_update_lut(params, &state.f_self, &state.v_f, diff, a_syn, lut_learn, lut_resolution, lut_max_idx);

        // 3. 代謝 (Metabolism)
        update_fatigue(params, &state.fatigue, state.amplitude, force);
        
        // 膠着判定
        if (std::abs(force) < params.dead_band) {
            state.inactivity_count++;
        } else {
            state.inactivity_count = 0;
        }

        // 4. 転生 (Rebirth)
        return try_rebirth(state, next_random_f, params);
    }
    return false;
}

// LUTを用いた高速ガウス励起
inline void RSTNNode::gaussian_excitation_lut(
    const RSTNParams& params, 
    double* p_amp, 
    double diff_f, 
    double a_syn, 
    const double* lut, 
    double resolution, 
    int max_idx
) {
    // インデックス計算: abs(diff) * resolution
    double abs_diff = std::abs(diff_f);
    int idx = static_cast<int>(abs_diff * resolution);
    
    // 範囲外チェックとクリッピング
    if (idx > max_idx) idx = max_idx; // 遠すぎる場合は最小値(0に近い値)を利用

    double efficiency = lut[idx];
    
    double target = a_syn * efficiency;
    *p_amp = std::min(target, params.a_limit);
}

// LUTを用いたRFA
inline double RSTNNode::rfa_update_lut(
    const RSTNParams& params, 
    double* p_f_self, 
    double* p_v_f, 
    double diff_f, 
    double a_syn, 
    const double* lut, 
    double resolution, 
    int max_idx
) {
    double force = 0.0;
    double abs_diff = std::abs(diff_f);

    if (abs_diff >= params.dead_band) {
        int idx = static_cast<int>(abs_diff * resolution);
        if (idx > max_idx) idx = max_idx;

        double learn_efficiency = lut[idx];
        force = (diff_f > 0 ? 1.0 : -1.0) * (a_syn * learn_efficiency);
        
        // エイジング (事前計算済み係数)
        force *= params.current_learning_rate;
    }
    
    double new_v_f = ((*p_v_f) * params.inertia) + (force * (1.0 - params.inertia));
    new_v_f *= params.viscosity;
    
    *p_v_f = new_v_f;
    *p_f_self += new_v_f;
    
    return force;
}

// 疲労計算 (変更なし)
inline void RSTNNode::update_fatigue(const RSTNParams& params, double* p_fatigue, double amplitude, double force) {
    if (std::abs(force) < params.dead_band) {
        *p_fatigue += params.c_load * (amplitude / params.a_limit);
    } else {
        *p_fatigue = std::max(0.0, *p_fatigue - params.c_recover);
    }
    if (amplitude < params.a_threshold) {
        *p_fatigue = std::max(0.0, *p_fatigue - params.c_recover);
    }
}

// 転生ロジック (変更なし)
inline bool RSTNNode::try_rebirth(RSTNState& state, double next_random_f, const RSTNParams& params) {
    double current_limit = state.fatigue_limit * params.current_limit_multiplier;
    bool is_overwork = (state.fatigue > current_limit);
    bool is_stagnant = (state.inactivity_count > params.inactivity_limit) && (state.amplitude < params.a_threshold);

    if (is_overwork || is_stagnant) {
        state.f_self = next_random_f;
        state.fatigue = 0.0;
        state.v_f = 0.0;
        state.amplitude = 0.0;
        state.inactivity_count = 0;
        return true;
    }
    return false;
}