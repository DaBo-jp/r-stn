#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "RSTNBox.hpp"
#include "RSTNParams.hpp"
#include "RSTNState.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rstn_cpp, m) {
    m.doc() = "R-STN C++ Core Module optimized for N^3 scale with AoS memory layout";

    // ------------------------------------------------------------------
    // RSTNParams のバインディング
    // ------------------------------------------------------------------
    py::class_<RSTNParams>(m, "RSTNParams")
        .def(py::init<>()) // デフォルトコンストラクタ
        
        // 物理定数
        .def_readwrite("sigma_ex", &RSTNParams::sigma_ex)
        .def_readwrite("sigma_learn", &RSTNParams::sigma_learn)
        .def_readwrite("inertia", &RSTNParams::inertia)
        .def_readwrite("viscosity", &RSTNParams::viscosity)
        .def_readwrite("dead_band", &RSTNParams::dead_band)
        .def_readwrite("c_load", &RSTNParams::c_load)
        .def_readwrite("c_recover", &RSTNParams::c_recover)
        .def_readwrite("a_threshold", &RSTNParams::a_threshold)
        .def_readwrite("a_limit", &RSTNParams::a_limit)
        
        // 減衰率 (新規追加)
        .def_readwrite("attenuation", &RSTNParams::attenuation)
        
        // 初期化・転生範囲
        .def_readwrite("f_min", &RSTNParams::f_min)
        .def_readwrite("f_max", &RSTNParams::f_max)
        .def_readwrite("fatigue_lim_min", &RSTNParams::fatigue_lim_min)
        .def_readwrite("fatigue_lim_max", &RSTNParams::fatigue_lim_max)

        // v2.0 エイジング & 代謝パラメータ
        .def_readwrite("max_steps", &RSTNParams::max_steps)
        .def_readwrite("p_critical", &RSTNParams::p_critical)
        .def_readwrite("p_mature", &RSTNParams::p_mature)
        .def_readwrite("decay_alpha", &RSTNParams::decay_alpha)
        .def_readwrite("growth_beta", &RSTNParams::growth_beta)
        .def_readwrite("inactivity_limit", &RSTNParams::inactivity_limit)

        // 動的係数 (参照用)
        .def_readonly("current_learning_rate", &RSTNParams::current_learning_rate)
        .def_readonly("current_limit_multiplier", &RSTNParams::current_limit_multiplier)
        
        // 内部値更新用
        .def("update_derived", &RSTNParams::update_derived);

    // ------------------------------------------------------------------
    // RSTNBox のバインディング
    // ------------------------------------------------------------------
    py::class_<RSTNBox>(m, "RSTNBox")
        .def(py::init<int, int>(), py::arg("n"), py::arg("seed") = 42)
        
        // 物理シミュレーション実行
        .def("step", &RSTNBox::step, py::arg("inputs"), py::arg("is_learning") = true)
        
        // 状態強制リセット
        .def("reset_states", &RSTNBox::reset_states)
        
        // LUTの再計算トリガー
        .def("update_tables", &RSTNBox::update_tables)
        
        // パラメータオブジェクトへの参照を返す (box.params でアクセス可能)
        .def_property_readonly("params", &RSTNBox::get_params, py::return_value_policy::reference)
        
        // Boxサイズ取得
        .def("get_size", &RSTNBox::get_size)

        // ------------------------------------------------------------------
        // ゼロコピー NumPy アクセサ (AoS View)
        // ------------------------------------------------------------------
        
        // 周波数 (f_self) のビューを取得
        .def("get_frequencies", [](RSTNBox& self) {
            RSTNState* ptr = self.get_states_ptr();
            return py::array_t<double>(
                {self.get_total_nodes()},             // shape: (N^3,)
                {sizeof(RSTNState)},                  // strides: 構造体1個分のバイト数
                &ptr->f_self,                         // data: 先頭要素の f_self アドレス
                py::cast(self)                        // base: 親オブジェクト(self)の寿命管理
            );
        })

        // 振幅 (amplitude) のビューを取得
        .def("get_amplitudes", [](RSTNBox& self) {
            RSTNState* ptr = self.get_states_ptr();
            return py::array_t<double>(
                {self.get_total_nodes()},
                {sizeof(RSTNState)},
                &ptr->amplitude,
                py::cast(self)
            );
        })

        // 疲労度 (fatigue) のビューを取得
        .def("get_fatigue", [](RSTNBox& self) {
            RSTNState* ptr = self.get_states_ptr();
            return py::array_t<double>(
                {self.get_total_nodes()},
                {sizeof(RSTNState)},
                &ptr->fatigue,
                py::cast(self)
            );
        });
}