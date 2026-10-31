#!/usr/bin/env bash
# 第9回の実験をまとめて回す（GPU 1枚・T4 16GB で動く設定）。結果は ./results/
# 教える組の数 0・25・50・100・150・200 × 方言3つ。T4 で約20分の見込み（README の「使い方」）
set -euo pipefail
cd "$(dirname "$0")"
# 借りたGPU環境に入っている古い torchao（0.10）が peft と合わず止まるので外す（無くても LoRA は動く）
python3 -m pip uninstall -y -q torchao >/dev/null 2>&1 || true
python3 net_dialect.py --pairs data/pairs.csv --sizes 0,25,50,100,150,200 --max_new 320 --out results/
python3 plot.py --rates results/bleed_rates.csv --out results/
