R-STN (Resonance-based Spatio-Temporal Network) プロジェクトの C++ コア実装用 `README.md` です。
Python (pybind11) からの利用と、C++ ネイティブでの利用の両方をカバーしています。

---

# R-STN: Resonance-based Spatio-Temporal Network (C++ Core)

**R-STN Theory Ver 1.10** に基づく、大規模物理シミュレーションエンジンの C++ 実装です。
従来のニューラルネットワーク（重み更新型）とは異なり、波の干渉と共鳴、および活動依存性の代謝（Rebirth）による自己組織化を行います。

本実装は  ( ノード) 以上の大規模系における「創発」現象を高速に観測するために設計されており、SoA (Structure of Arrays) メモリレイアウトと OpenMP 並列化により、Python プロトタイプ比で数十倍〜数百倍の演算速度を実現します。

## 主な特徴

* **SoA (Structure of Arrays) アーキテクチャ**: ノードデータを巨大な連続配列で管理し、CPU キャッシュ効率を最大化。
* **OpenMP 並列化**: 物理演算ループを全 CPU コアで並列実行。
* **ゼロコピー Python バインディング**: C++ 側のメモリ領域を NumPy 配列として直接参照可能。データのコピーコストなしで高速な可視化が可能。
* **動的パラメータ調整**: シミュレーション実行中に物理定数（慣性、粘性、共鳴帯域など）を Python/C++ 側から動的に変更可能。

## 必要要件

* **C++ コンパイラ**: C++17 以上をサポートするもの (GCC, Clang, MSVC)
* **OpenMP**: 並列計算用ライブラリ (`libomp` 等)
* **Python**: 3.8 以上 (Python バインディング利用時)
* **ライブラリ**:
* `pybind11` (Python バインディング用)
* `numpy`



## ビルド方法

### 1. Python モジュールとしてビルド (推奨)

Python から高速演算エンジンとして利用する場合です。

```bash
# 依存ライブラリのインストール
pip install pybind11 numpy

# ビルド (現在のディレクトリに .so または .pyd を生成)
python3 setup.py build_ext --inplace

```

### 2. C++ ネイティブアプリとしてビルド

C++ のみで完結するシミュレーションや、さらなるパフォーマンスチューニングを行う場合です。

```bash
# GCC / Linux の例
g++ -O3 -fopenmp -std=c++17 main.cpp RSTNBox.cpp RSTNNode.cpp -o rstn_sim

# 実行
./rstn_sim

```

---

## 使い方: Python から利用する場合

`rstn_cpp` モジュールをインポートして使用します。計算結果は NumPy 配列として直接取得できます。

```python
import rstn_cpp
import numpy as np
import time

# 1. Boxの初期化 (N=32 -> 32^3 = 32,768 ノード)
N = 32
box = rstn_cpp.RSTNBox(N, seed=42)

# 2. パラメータの動的調整 (Box.params 経由でアクセス)
# シミュレーションの途中で書き換え可能
box.params.inertia = 0.99       # 慣性を上げてスローモーションに
box.params.viscosity = 0.2      # 粘性を下げて反応を良くする
box.params.sigma_ex = 15.0      # 共鳴帯域を絞る

print(f"Total Nodes: {box.get_total_nodes()}")

# 3. 入力データの作成
# list of tuples: (node_index, (amplitude, frequency))
inputs = []
# 例: Z=0 面 (入力面) に 25Hz の信号を与える
for y in range(N):
    for x in range(N):
        idx = x + y * N  # Z=0
        inputs.append((idx, (100.0, 25.0)))

# 4. シミュレーションループ
start = time.time()
for step in range(400):
    # step(inputs, is_learning)
    # is_learning=True で適応・代謝が有効化
    box.step(inputs, is_learning=True)

    if step % 100 == 0:
        # 5. データの取得 (ゼロコピー)
        # C++のメモリを直接参照しているため非常に高速
        freqs = box.get_frequencies()  # numpy array (read-only view)
        amps = box.get_amplitudes()
        print(f"Step {step}: Max Amp = {np.max(amps):.2f}")

print(f"Elapsed: {time.time() - start:.4f} sec")

```

---

## 使い方: C++ から利用する場合

ヘッダファイルをインクルードし、`RSTNBox` クラスを使用します。

```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include "RSTNBox.hpp"

int main() {
    // 1. Boxの初期化
    const int N = 32;
    RSTNBox box(N, 42);

    // 2. パラメータ調整
    box.get_params().inertia = 0.95;
    box.get_params().sigma_learn = 20.0;

    // 3. 入力データの作成
    // vector<pair<int, pair<double, double>>>
    // {index, {amplitude, frequency}}
    std::vector<std::pair<int, std::pair<double, double>>> inputs;
    inputs.reserve(N * N);
    
    // Z=0 面への入力
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int idx = x + y * N;
            inputs.push_back({idx, {100.0, 25.0}});
        }
    }

    // 4. シミュレーションループ
    std::cout << "Starting simulation..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < 400; ++s) {
        // step(inputs, is_learning)
        box.step(inputs, true);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Time: " << diff.count() << " s" << std::endl;

    // データへのアクセス (ポインタ経由)
    double* freqs = box.get_f_self_ptr();
    std::cout << "Node 0 Freq: " << freqs[0] << std::endl;

    return 0;
}

```

## アーキテクチャ

* **RSTNNode (Stateless Kernel)**:
* 状態を持ちません。与えられたメモリアドレスに対して物理演算（共鳴、RFA、代謝）を行う純粋なロジッククラスです。


* **RSTNBox (Memory Manager)**:
* 巨大な `std::vector` (SoA) を保持・管理します。
* OpenMP を使用して `step` 関数内で並列計算を制御します。
* `RSTNParams` を保持し、ノードの振る舞いを一括制御します。


* **RSTNParams**:
* 物理定数（ 等）を管理する構造体です。Box が保持し、Node に参照渡しされます。



## ライセンス

MIT License