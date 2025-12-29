# R-STN: Resonance-Based Spatiotemporal Network

**自己組織化知能のための物理駆動型アーキテクチャ**

R-STN (Resonance-Based Spatiotemporal Network) は、従来のニューラルネットワークにおける「重み」の概念を完全に排除し、3次元時空間内での「波動干渉」と「共鳴」を物理的にシミュレートすることで情報を処理する、全く新しい知能アーキテクチャです。

## コア・コンセプト

*   **重みなき物理 (Weightless Physics):** 行列演算による「重み」は存在しません。論理は、位相の遅延と、共鳴による「経路（トンネル）」の形成・トポロジー変化のみによって構築されます。
*   **3次元時空間ダイナミクス:** 信号は3次元格子状に配置されたノード間を波動として伝播します。
*   **発達段階とエイジング (Aging):** 学習プロセスに「ライフサイクル（臨界期・成熟期・安定期）」を導入。時間経過とともに学習率や代謝パラメータがパレート則に従って動的に変化します。
*   **高効率C++エンジン:** OpenMPによる並列化と、LUT（Look-Up Table）化された物理演算により、高速なシミュレーションを実現しています。

## ディレクトリ構成

```text
.
├── docs/                  # 理論ドキュメント、仕様書
├── rstn/                  # コアパッケージ
│   ├── cpp_src/           # C++ 物理エンジンソース (OpenMP対応)
│   ├── box.py             # RSTNBox のPythonラッパー
│   └── node.py            # RSTNNode のPythonラッパー
├── experiments/           # 実験・シミュレーション
│   ├── cases/             # シナリオ定義 (トンネリング、メモリ等)
│   ├── runners/           # 実行スクリプト
│   ├── visualization/     # 可視化ツール
│   └── data/              # 実験データ出力先 (Git管理外)
├── setup.py               # ビルド・インストール設定
└── pyproject.toml         # ビルドシステム設定
```

## インストール方法

C++17およびOpenMPに対応したコンパイラが必要です。

```bash
# 1. 仮想環境の構築
python3 -m venv venv
source venv/bin/activate

# 2. 依存ライブラリのインストールとビルド
pip install --upgrade pip
pip install -e .
```

## シミュレーションの実行

ターゲットの切り替えと経路形成を確認する実験（Case 6）を実行します：

```bash
python experiments/runners/run_cpp_sim.py
```

結果は `experiments/data/cpp_output/Case6_Discrete.npz` に保存されます。

## 可視化

シミュレーション結果を3次元プロットとして画像化します：

```bash
python experiments/visualization/visualize_cpp.py
```

`experiments/data/cpp_output/` 内に連番画像が生成されます。

## ライセンス

[MIT License](LICENSE)
