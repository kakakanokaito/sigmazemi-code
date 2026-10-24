# 第3回 v4 の追加実験：なぜ「男→女」と「王→女王」の矢印がそろうのか（シグマゼミ、2026-10-24）
# 考え方は GloVe の論文（Pennington ほか 2014）の表1「共起の比」と同じ。
#   P(k | w) ＝ 単語 w の前後5語に、単語 k が出てきた割合
#   比 ＝ P(k | king) ÷ P(k | queen)  と  P(k | man) ÷ P(k | woman) をくらべる
# 使う文章：text8（英語版ウィキペディアの最初の約1,700万語）
#   入手：https://github.com/RaRe-Technologies/gensim-data/releases/download/text8/text8.gz
#   使い方：python3 cooccur_ratio.py <text8.gz のパス>
import gzip, sys
from collections import Counter

WIN = 5
CENTERS = ["king", "queen", "man", "woman"]
PROBES = ["he", "his", "him", "she", "her", "husband", "wife", "son", "daughter",
          "throne", "crown", "royal", "reign", "the", "of", "water", "computer"]

with gzip.open(sys.argv[1], "rt") as f:
    toks = f.read().split()
print(f"文章の長さ：{len(toks):,} 語")

ctx = {c: Counter() for c in CENTERS}
cnt = Counter()
for i, t in enumerate(toks):
    if t in ctx:
        cnt[t] += 1
        lo, hi = max(0, i - WIN), min(len(toks), i + WIN + 1)
        ctx[t].update(toks[lo:i] + toks[i + 1:hi])
for c in CENTERS:
    print(f"  {c:6s} は {cnt[c]:,} 回")

def P(k, w):
    return ctx[w][k] / sum(ctx[w].values())

print(f"\nP(k|w) は前後{WIN}語に k が出た割合（×1000 で表示）")
print(f"{'k':10s} {'king':>7s} {'queen':>7s} {'王÷女王':>8s} | {'man':>7s} {'woman':>7s} {'男÷女':>8s}")
for k in PROBES:
    a, b, c, d = (P(k, w) * 1000 for w in CENTERS)
    r1 = a / b if b else float("inf")
    r2 = c / d if d else float("inf")
    print(f"{k:10s} {a:7.2f} {b:7.2f} {r1:8.2f} | {c:7.2f} {d:7.2f} {r2:8.2f}")
