#pragma once

struct RSTNState {
    double f_self;          // 固有周波数
    double amplitude;       // 振幅
    double v_f;             // 周波数速度
    double fatigue;         // 疲労度
    double fatigue_limit;   // 疲労限界
    int inactivity_count;   // 不活動カウンタ (for Inactivity Death)
};