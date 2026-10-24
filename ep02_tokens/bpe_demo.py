# 第2回 実験2：トークンの区切り方を自分で作る（BPE：一番よく並ぶ2つをくっつける、を繰り返す）
# 使い方: python3 bpe_demo.py ../ep01_next_token/wagahaiwa_nekodearu.txt   （標準ライブラリだけで動く）
import sys
from collections import Counter
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ep01_next_token"))
from next_char import load   # 第1回と同じ読み込み方（ルビ・注記を消す）

text = load(sys.argv[1])[:100000]          # 最初の10万文字だけ使う（時間短縮）
seq = list(text)                           # 最初は1文字＝1トークン
print(f"はじめ: {len(seq):,}トークン")

for step in range(1, 301):
    pairs = Counter(zip(seq, seq[1:]))
    (a, b), n = pairs.most_common(1)[0]    # 一番よく並んでいる2つ
    merged, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            merged.append(a + b); i += 2
        else:
            merged.append(seq[i]); i += 1
    seq = merged
    if step <= 20 or step % 50 == 0:
        print(f"{step:3d}回目: 「{a}」+「{b}」→「{a+b}」({n}回)  全体 {len(seq):,}トークン")

print("\n冒頭をトークンで区切ると:")
print(" | ".join(seq[:40]))
