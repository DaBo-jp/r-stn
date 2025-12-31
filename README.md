# R-STN: Resonance-Based Spatiotemporal Network

**自己組織化知能のための物理駆動型アーキテクチャ**

R-STN (Resonance-Based Spatiotemporal Network) は、従来のニューラルネットワークにおける「重み」の概念を排除し、3次元時空間内での「波動干渉」と「共鳴」を物理的にシミュレートすることで情報を処理する知能アーキテクチャです。

## クイックスタートガイド

リポジトリをクローンしてから最初のシミュレーションを実行するまでの手順です。

### 1. 前提条件 (Prerequisites)

以下のシステム要件を満たしている必要があります。

*   **OS:** Linux (Ubuntu/Debian推奨) または macOS
*   **コンパイラ:** C++17 対応かつ OpenMP をサポートするコンパイラ (`g++` または `clang++`)
*   **Python:** 3.8 以上

#### Ubuntu/Debian の場合
```bash
sudo apt update
sudo apt install build-essential python3-dev python3-venv git
```

### 2. 環境構築 (Installation)

```bash
# リポジトリのクローン
git clone https://github.com/mizinko-kinako/r-stn.git
cd r-stn/mono-node-research/sim

# 仮想環境の作成と有効化
python3 -m venv venv
source venv/bin/activate

# 依存ライブラリのインストール
pip install --upgrade pip
pip install -r requirements.txt

# C++エンジンのビルド (rstn_cpp モジュールの生成)
# ※この手順で .so ファイルが生成されます
pip install -e .
```

### 3. 動作確認 (Verification)

環境が正しく構築されたか確認するために、バリデーションスクリプトを実行します。
これにより、基本的な4つの実験ケースが実行され、結果が可視化されます。

```bash
./run_box_validation.sh
```

実行が完了すると、`reports/` ディレクトリに以下のファイルが生成されます：
*   実験データのログ (`.npz`ファイル)
*   3D可視化画像 (`.png`)

---

## ディレクトリ構成

```text
.
├── docs/                  # 理論ドキュメント、仕様書
├── rstn/                  # コアパッケージ
│   └── cpp_src/           # C++ 物理エンジンソース (OpenMP対応)
├── experiments/           # 実験・シミュレーション
│   ├── cases/             # シナリオ定義 (トンネリング、メモリ等)
│   ├── runners/           # 実行スクリプト (バッチ実行、スイープ等)
│   ├── visualization/     # 可視化・解析ツール
│   └── data/              # 実験データ出力先 (Git管理外)
├── reports/               # 可視化結果 (画像・動画) の出力先 (Git管理外)
├── setup.py               # ビルド・インストール設定
├── requirements.txt       # Python依存ライブラリ
└── pyproject.toml         # ビルドシステム設定
```

## 高度な使用法

### ターゲット切り替え実験 (Case 6)
動的なターゲット変更に対する経路の再形成（ダイナミック・ルーティング）を観察します。

```bash
python experiments/runners/run_cpp_sim.py
```

### パラメータスイープ
広範囲な物理パラメータの組み合わせ（粘性、慣性、減衰率など）を探索し、最適な「脳」の条件を探します。

```bash
python experiments/runners/run_sweep_complex.py
```

### 結果の可視化と動画生成
特定の実験データを詳細に解析したり、動画として出力したりする場合：

```bash
# 特定のデータファイルを3D動画化 (要ffmpeg)
python experiments/visualization/visualize_movie.py
```

※ `visualize_movie.py` は、`experiments/data/cpp_output/` 内にある `Case5` または `Case6` のデータを自動的に探して処理します。

## 注意事項
- **Pythonパス:** 全てのスクリプトは `sim` ディレクトリ内で実行することを想定しています。
- **ffmpeg:** 動画生成機能を使用する場合、システムに `ffmpeg` がインストールされていることが推奨されます（ない場合はGIFアニメーションが生成されます）。

## ライセンス

[MIT License](LICENSE)