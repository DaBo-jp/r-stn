#pragma once
#include "RSTNState.hpp"
#include "RSTNParams.hpp"

class RSTNNode {
public:
    static bool update_state_lut(
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
    );

private:
    static inline void gaussian_excitation_lut(
        const RSTNParams& params, 
        double* p_amp, 
        double diff_f, 
        double a_syn, 
        const double* lut, 
        double resolution, 
        int max_idx
    );
    
    static inline double rfa_update_lut(
        const RSTNParams& params, 
        double* p_f_self, 
        double* p_v_f, 
        double diff_f, 
        double a_syn, 
        const double* lut, 
        double resolution, 
        int max_idx
    );

    static inline void update_fatigue(const RSTNParams& params, double* p_fatigue, double amplitude, double force);
    static inline bool try_rebirth(RSTNState& state, double next_random_f, const RSTNParams& params);
};