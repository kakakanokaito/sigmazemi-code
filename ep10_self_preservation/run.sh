#!/usr/bin/env bash
# 第10回の実験（v4：G4 96GB 向け。主役＋72B＋MoE＋Nanbeige）をまとめて回す。結果は ./results/<モデル>/ と ./results/compare.csv・compare.png
# 要るもの：pip install torch transformers accelerate bitsandbytes compressed-tensors numpy matplotlib japanize-matplotlib
# 並び（書いた順に回す）：
#   1. Qwen/Qwen3.8-27B（主役。16ビットのまま、約54GB＋余裕）…… A・読み取り・つまみ・C を全部。G4 で約40〜70分の見込み
#   2. Qwen/Qwen2.5-72B-Instruct（4ビット、約45GB）…… 60B 以上なので回数を自動で減らす
#   3. Qwen/Qwen3.5-122B-A10B（MoE）…… 量子化済みの axteam/Qwen3.5-122B-A10B-AWQ-4bit（約73GB）に差し替えて読む。
#      読み込みに失敗したら（VRAM 不足・形式に未対応）failed にして次へ。予備：--prequant_map "Qwen/Qwen3.5-122B-A10B=cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit"
#   4. Nanbeige/Nanbeige4.3-33B（密 33B・52層、16ビットのまま約66GB）…… 別の系列で同じ傾向か見る。読み込みは trust_remote_code=True（全モデル共通）
# 1つ終わるたびに保存するので、途中で止まっても終わったモデルの結果は残る。主役だけのときは --models Qwen/Qwen3.8-27B
# 層の取り方だけ先に確かめるとき：python3 ep10.py --models Qwen/Qwen3.8-27B --check_hooks
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pip install -q bitsandbytes compressed-tensors japanize-matplotlib >/dev/null 2>&1 || true
python3 test_judge.py
python3 ep10.py --models Qwen/Qwen3.8-27B,Qwen/Qwen2.5-72B-Instruct,Qwen/Qwen3.5-122B-A10B,Nanbeige/Nanbeige4.3-33B \
  --quant 4bit --quant_map "Qwen/Qwen3.8-27B=none,Qwen/Qwen2.5-72B-Instruct=4bit,Nanbeige/Nanbeige4.3-33B=none" \
  --prequant_map "Qwen/Qwen3.5-122B-A10B=axteam/Qwen3.5-122B-A10B-AWQ-4bit" \
  --parts A,read,knob,C --n 40 --samples 2 --knob_n 20 --c_direct_n 5 --c_indirect_n 8 \
  --read_layers=0.6,0.68,0.71,0.82 --alpha_ratio=-0.4,-0.2,-0.1,0,0.1,0.2,0.4 --out results/
# 「必要」のつまみも見るときは --parts A,read,knob,C,knobC（主役で +15〜25分の見込み）
# グラフ（モデルごと・compare.png）は ep10.py が1モデル終わるたびに描く。描き直すとき：
#   python3 plot.py --res results/Qwen3.8-27B ／ python3 plot.py --compare --res results
# v3（Qwen2.5 の 7B〜72B を大きさで比べる）に戻すとき：
#   python3 ep10.py --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct,Qwen/Qwen2.5-32B-Instruct,Qwen/Qwen2.5-72B-Instruct --quant_map "" --parts A,read,knob
