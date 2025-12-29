#include "RSTNBox.hpp"
#include "RSTNNode.hpp"
#include <stdexcept>
#include <iostream>
#include <cstring> // memset用
#include <cmath>   // std::abs用

RSTNBox::RSTNBox(int n, int seed) : N(n), current_step(0) {
    if (n <= 0 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("Size N must be a power of 2.");
    }
    total_nodes = static_cast<size_t>(n) * n * n;

    // メモリ確保
    states = std::make_unique<RSTNState[]>(total_nodes);
    prev_amp = std::make_unique<double[]>(total_nodes);
    prev_f = std::make_unique<double[]>(total_nodes);
    random_pool = std::make_unique<double[]>(total_nodes);

    // 入力バッファ確保
    input_map_amp = std::make_unique<double[]>(total_nodes);
    input_map_freq = std::make_unique<double[]>(total_nodes);
    input_map_active = std::make_unique<bool[]>(total_nodes);

    // RNG初期化
    int max_threads = omp_get_max_threads();
    thread_rngs.resize(max_threads);
    for(int i=0; i<max_threads; ++i) {
        thread_rngs[i].seed(seed + i);
    }

    // LUTの初期化
    update_tables();
    reset_states();
}

void RSTNBox::update_tables() {
    // 1. パラメータの内部係数更新
    m_params.update_derived();

    // 2. ガウス関数LUTの事前計算
    lut_ex.resize(LUT_SIZE);
    lut_learn.resize(LUT_SIZE);
    
    // diff は 0.0 から刻み幅(1/LUT_RESOLUTION)ずつ増加
    double step_val = 1.0 / (double)LUT_RESOLUTION;
    
    #pragma omp parallel for
    for (int i = 0; i < LUT_SIZE; ++i) {
        double diff = i * step_val;
        double diff_sq = diff * diff;
        
        // Excitation Curve
        lut_ex[i] = std::exp(diff_sq * m_params._coeff_ex);
        
        // Learning Q-Curve
        lut_learn[i] = std::exp(diff_sq * m_params._coeff_learn);
    }

    // 3. エイジングスケジュールの事前計算
    size_t schedule_size = static_cast<size_t>(m_params.max_steps) + 1;
    schedule_lr.resize(schedule_size);
    schedule_limit.resize(schedule_size);

    #pragma omp parallel for
    for (size_t s = 0; s < schedule_size; ++s) {
        double p = static_cast<double>(s) / static_cast<double>(m_params.max_steps);
        
        // Learning Rate Decay (Pareto)
        schedule_lr[s] = 1.0 / std::pow(1.0 + p / m_params.p_critical, m_params.decay_alpha);
        
        // Fatigue Limit Growth (Pareto)
        schedule_limit[s] = std::pow(1.0 + p / m_params.p_mature, m_params.growth_beta);
    }
}

void RSTNBox::reset_states() {
    current_step = 0; // Reset aging
    m_params.current_learning_rate = schedule_lr[0];
    m_params.current_limit_multiplier = schedule_limit[0];

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::uniform_real_distribution<double> dist_f(m_params.f_min, m_params.f_max);
        std::uniform_real_distribution<double> dist_lim(m_params.fatigue_lim_min, m_params.fatigue_lim_max);

        #pragma omp for
        for (size_t i = 0; i < total_nodes; ++i) {
            states[i].f_self = dist_f(thread_rngs[tid]);
            states[i].fatigue_limit = dist_lim(thread_rngs[tid]);
            states[i].amplitude = 0.0;
            states[i].v_f = 0.0;
            states[i].fatigue = 0.0;
            states[i].inactivity_count = 0;
            random_pool[i] = dist_f(thread_rngs[tid]);
        }
    }
}

void RSTNBox::step(const std::vector<std::pair<int, std::pair<double, double>>>& inputs, bool is_learning) {
    
    // --- Phase 0: エイジング更新 (LUT参照による高速化) ---
    if (is_learning) {
        if (current_step < (long long)schedule_lr.size() - 1) {
            current_step++;
            m_params.current_learning_rate = schedule_lr[current_step];
            m_params.current_limit_multiplier = schedule_limit[current_step];
        } else {
            // max_steps超過時は最終値を維持
            m_params.current_learning_rate = schedule_lr.back();
            m_params.current_limit_multiplier = schedule_limit.back();
        }
    }

    // --- Phase 0.5: 入力データの高速マッピング ---
    std::memset(input_map_active.get(), 0, total_nodes * sizeof(bool));
    
    for (const auto& inp : inputs) {
        int idx = inp.first;
        input_map_amp[idx] = inp.second.first;
        input_map_freq[idx] = inp.second.second;
        input_map_active[idx] = true;
    }

    // --- Phase 1: バッファリング & 乱数生成 ---
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::uniform_real_distribution<double> dist_f(m_params.f_min, m_params.f_max);

        #pragma omp for
        for (size_t i = 0; i < total_nodes; ++i) {
            prev_amp[i] = states[i].amplitude;
            prev_f[i] = states[i].f_self;
            random_pool[i] = dist_f(thread_rngs[tid]);
        }
    }
    
    // LUT用ポインタ取得 (OpenMP内での参照用)
    const double* p_lut_ex = lut_ex.data();
    const double* p_lut_learn = lut_learn.data();
    const double lut_res = (double)LUT_RESOLUTION;
    const int lut_max_idx = LUT_SIZE - 1;

    // --- Phase 2: 物理演算ループ (Spatial Filtering & Physics) ---
    #pragma omp parallel for
    for (int i = 0; i < (int)total_nodes; ++i) {
        // 座標計算
        int x = i % N;
        int y = (i / N) % N;
        int z = i / (N * N);

        // 集計用変数
        double w_f_sum = 0.0; // 周波数の重み付き和
        double w_a_sum = 0.0; // 振幅の単純和（平均計算用）
        int neighbor_count = 0; // 近傍数

        // ラムダ関数: 近傍情報の収集
        auto add_neighbor = [&](int ni) {
            double a = prev_amp[ni];
            double f = prev_f[ni];
            double abs_a = std::abs(a);
            
            w_a_sum += abs_a;       // 振幅の合計
            w_f_sum += abs_a * f;   // 周波数の加重合計
            neighbor_count++;
        };

        // 6近傍アクセス
        if (x > 0)   add_neighbor(i - 1);
        if (x < N-1) add_neighbor(i + 1);
        if (y > 0)   add_neighbor(i - N);
        if (y < N-1) add_neighbor(i + N);
        if (z > 0)   add_neighbor(i - N*N);
        if (z < N-1) add_neighbor(i + N*N);

        double a_syn = 0.0;
        double f_syn = 0.0;

        // 入力がある場合とない場合で処理を分岐
        if (input_map_active[i]) {
            // 直接入力 (Attenuationなし、純粋な入力値)
            a_syn = std::abs(input_map_amp[i]);
            f_syn = input_map_freq[i];
        } else {
            // 空間伝播 (Spatial Propagation)
            double avg_amp = (neighbor_count > 0) ? (w_a_sum / neighbor_count) : 0.0;
            
            // 減衰 (Attenuation)
            a_syn = avg_amp * (1.0 - m_params.attenuation);
            
            // 周波数の重心計算
            f_syn = (w_a_sum > 1e-9) ? (w_f_sum / w_a_sum) : prev_f[i];
        }

        // 状態更新 (RSTNNodeへ委譲 - LUT版)
        RSTNNode::update_state_lut(
            m_params,
            states[i],
            a_syn,
            f_syn,
            random_pool[i],
            is_learning,
            p_lut_ex,
            p_lut_learn,
            lut_res,
            lut_max_idx
        );
    }
}