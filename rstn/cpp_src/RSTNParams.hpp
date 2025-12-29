#pragma once
#include <cmath>

struct RSTNParams {
    // 物理定数
    double sigma_ex = 10.0;      // 共鳴帯域 (狭く設定)
    double sigma_learn = 20.0;
    double inertia = 0.95;
    double viscosity = 0.5;
    double dead_band = 1.0;
    double c_load = 10.0;
    double c_recover = 15.0;
    double a_threshold = 1.0;
    double a_limit = 100.0;

    // 空間減衰率
    double attenuation = 0.15;

    // 初期化・転生範囲
    double f_min = -40.0;
    double f_max = 40.0;
    double fatigue_lim_min = 900.0;
    double fatigue_lim_max = 1100.0;

    // v2.0 エイジング & 代謝パラメータ
    long long max_steps = 10000;    // ライフサイクル全ステップ
    double p_critical = 0.05;       // 臨界期終了 (5%)
    double p_mature = 0.33;         // 成熟期終了 (33%)
    double decay_alpha = 2.0;       // 学習率減衰係数
    double growth_beta = 2.0;       // 疲労耐性成長係数
    int inactivity_limit = 100;     // 不活動死のリミット

    // 動的係数 (Box::step で更新される)
    double current_learning_rate = 1.0;   // エイジングによる学習率係数
    double current_limit_multiplier = 1.0;// エイジングによる疲労限界倍率

    // 内部計算用
    double _coeff_ex = -1.0 / (2.0 * 10.0 * 10.0);
    double _coeff_learn = -1.0 / (2.0 * 20.0 * 20.0);

    void update_derived() {
        _coeff_ex = -1.0 / (2.0 * sigma_ex * sigma_ex);
        _coeff_learn = -1.0 / (2.0 * sigma_learn * sigma_learn);
    }
};