#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include "RSTNBox.hpp"
#include "RSTNState.hpp"
#include "RSTNParams.hpp"

// 統計情報の表示用ヘルパー
void print_stats(int step, double elapsed_ms, RSTNBox& box, double input_freq, bool is_learning) {
    RSTNState* states = box.get_states_ptr();
    size_t total = box.get_total_nodes();
    
    double max_amp = 0.0;
    double avg_amp = 0.0;
    double avg_fatigue = 0.0;
    int active_nodes = 0;

    // 簡易統計 (表示用なのでシングルスレッドでOK)
    for(size_t i=0; i<total; ++i) {
        double amp = states[i].amplitude;
        if (amp > 1.0) active_nodes++;
        if (amp > max_amp) max_amp = amp;
        avg_amp += amp;
        avg_fatigue += states[i].fatigue;
    }
    avg_amp /= total;
    avg_fatigue /= total;

    // コンソール出力
    std::cout << "| " << std::setw(4) << step 
              << " | " << std::setw(9) << (is_learning ? "LEARNING" : "INFERENCE")
              << " | " << std::setw(6) << input_freq << " Hz"
              << " | " << std::setw(7) << std::fixed << std::setprecision(3) << elapsed_ms << " ms"
              << " | " << std::setw(6) << active_nodes 
              << " | " << std::setw(6) << std::setprecision(1) << max_amp 
              << " | " << std::setw(6) << std::setprecision(1) << avg_fatigue << " |" 
              << std::endl;
}

int main() {
    // --- 設定 ---
    const int N = 32;          // 32^3 = 32,768 ノード
    const int MAX_STEPS = 400; // シミュレーションステップ数
    const int LOG_INTERVAL = 20; // ログ出力間隔

    std::cout << "==============================================================" << std::endl;
    std::cout << " R-STN C++ Core Runner (N=" << N << ", Total Nodes=" << N*N*N << ")" << std::endl;
    std::cout << "==============================================================" << std::endl;

    // 1. Box初期化
    RSTNBox box(N, 42); 
    
    // パラメータ調整例: 慣性を少し上げる
    box.get_params().inertia = 0.95;
    box.get_params().update_derived(); // 派生値を更新

    // 入力データのバッファ (Z=0 面への入力)
    std::vector<std::pair<int, std::pair<double, double>>> inputs;
    inputs.reserve(N * N);

    auto start_total = std::chrono::high_resolution_clock::now();

    // ヘッダー表示
    std::cout << "| Step | Mode      | Input   | Compute | Active | MaxAmp | AvgFat |" << std::endl;
    std::cout << "|------|-----------|---------|---------|--------|--------|--------|" << std::endl;

    // --- シミュレーションループ ---
    for (int s = 0; s < MAX_STEPS; ++s) {
        
        // シナリオ (Case 4): 
        // 前半(0-199): 学習モード (20Hz)
        // 後半(200-):  推論モード (-40Hz, 学習OFF)
        bool is_learning = (s < 200);
        double target_freq = is_learning ? 20.0 : -40.0;

        // 入力データの生成 (Z=0 平面のみに入力)
        inputs.clear();
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                int idx = x + y * N; // Z=0
                inputs.push_back({idx, {100.0, target_freq}});
            }
        }

        // --- 物理演算実行 & 計測 ---
        auto t0 = std::chrono::high_resolution_clock::now();
        
        box.step(inputs, is_learning);
        
        auto t1 = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // --- ログ出力 ---
        if (s % LOG_INTERVAL == 0 || s == MAX_STEPS - 1) {
            print_stats(s, step_ms, box, target_freq, is_learning);
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_total - start_total).count();

    std::cout << "==============================================================" << std::endl;
    std::cout << " Simulation Finished." << std::endl;
    std::cout << " Total Time: " << total_sec << " s" << std::endl;
    std::cout << " Avg Speed : " << (MAX_STEPS / total_sec) << " steps/sec" << std::endl;
    std::cout << "==============================================================" << std::endl;

    return 0;
}