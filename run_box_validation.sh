#!/bin/bash
SIZE=32
# 全シミュレーション実行
python3 sim_case1_tunneling.py $SIZE
python3 sim_case2_territory.py $SIZE
python3 sim_case3_memory.py $SIZE
python3 sim_case4_inference.py $SIZE

# 全データ可視化
for f in *.npz; do
    python3 visualize_box.py "$f"
done

mkdir -p reports
mv *.png reports/
mv *.npz reports/
echo "Done. All results with precise compute times are in reports/"