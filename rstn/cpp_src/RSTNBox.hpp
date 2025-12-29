#pragma once

#include <memory>
#include <random>
#include <vector>
#include <omp.h>
#include "RSTNParams.hpp"
#include "RSTNState.hpp"

class RSTNBox {
private:
    int N;
    size_t total_nodes;
    RSTNParams m_params;
    long long current_step; // エイジング管理用ステップカウンタ

    // メモリ管理 (AoS)
    std::unique_ptr<RSTNState[]> states;

    // バッファ
    std::unique_ptr<double[]> prev_amp;
    std::unique_ptr<double[]> prev_f;
    std::unique_ptr<double[]> random_pool; 
    
    // 入力高速化マップ
    std::unique_ptr<double[]> input_map_amp;
    std::unique_ptr<double[]> input_map_freq;
    std::unique_ptr<bool[]>   input_map_active;
    
    std::vector<std::mt19937> thread_rngs;

    // --- 高速化用 Look-Up Tables (LUT) ---
    // 解像度: 周波数差 0.001 単位でマッピング
    static constexpr int LUT_RESOLUTION = 1000;
    static constexpr int LUT_SIZE = 100 * LUT_RESOLUTION; // 差分100.0までカバー
    
    std::vector<double> lut_ex;          // 励起用ガウス関数テーブル
    std::vector<double> lut_learn;       // 学習用ガウス関数テーブル
    
    // エイジングスケジュール (Step -> Value)
    std::vector<double> schedule_lr;     // 学習率スケジュール
    std::vector<double> schedule_limit;  // 疲労限界スケジュール

public:
    RSTNBox(int n, int seed = 42);

    void step(const std::vector<std::pair<int, std::pair<double, double>>>& inputs, bool is_learning);
    void reset_states();
    
    // パラメータ変更時にLUTを再計算する
    void update_tables();

    RSTNParams& get_params() { return m_params; }
    RSTNState* get_states_ptr() { return states.get(); }
    size_t get_total_nodes() const { return total_nodes; }
    int get_size() const { return N; }
};