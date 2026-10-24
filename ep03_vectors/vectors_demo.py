# 第3回「なぜ 王−男＋女≒女王 になるのか」の実験（シグマゼミ、2026-10-24）
# 使う単語ベクトル：GloVe（Stanford、Wikipedia 2014＋Gigaword 5 で学習、40万語×100次元）
#   入手：gensim の glove-wiki-gigaword-100（https://github.com/RaRe-Technologies/gensim-data）
#   使い方：python3 vectors_demo.py <glove-wiki-gigaword-100.gz のパス>
# 計算は numpy だけで書いている（今日の式 cos θ = a·b / (|a||b|) をそのまま使う）
import gzip, sys
import numpy as np

path = sys.argv[1]
words, vecs = [], []
with gzip.open(path, "rt", encoding="utf-8") as f:
    first = f.readline().split()
    if len(first) != 2:  # 先頭行が「語数 次元数」でなければ、それもデータ
        words.append(first[0]); vecs.append(np.array(first[1:], dtype=np.float32))
    for line in f:
        p = line.rstrip().split(" ")
        words.append(p[0]); vecs.append(np.array(p[1:], dtype=np.float32))
V = np.vstack(vecs)
idx = {w: i for i, w in enumerate(words)}
unit = V / np.linalg.norm(V, axis=1, keepdims=True)
print(f"語彙 {len(words):,} 語 × {V.shape[1]} 次元")


def vec(w): return V[idx[w]]


def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def nearest(v, k=5, exclude=()):
    s = unit @ (v / np.linalg.norm(v))
    out = []
    for i in np.argsort(-s):
        if words[i] in exclude: continue
        out.append((words[i], float(s[i])))
        if len(out) == k: break
    return out


print("\n① 1つの単語は100個の数字（先頭5個）")
for w in ["king", "queen", "cat"]:
    print(f"  {w:6s}", " ".join(f"{x:+.3f}" for x in vec(w)[:5]), "…")

print("\n② 似ている度合い cos（1に近いほど同じ向き）")
for a, b in [("cat", "dog"), ("cat", "kitten"), ("cat", "car"), ("king", "queen"), ("king", "banana"), ("strawberry", "blueberry"), ("strawberry", "democracy")]:
    print(f"  cos({a}, {b}) = {cos(vec(a), vec(b)):.3f}")

print("\n③ king − man + woman に近い単語")
q = vec("king") - vec("man") + vec("woman")
print("  入れた3語も候補に含めると：", ", ".join(f"{w} {s:.3f}" for w, s in nearest(q, 5)))
print("  入れた3語を除くと      ：", ", ".join(f"{w} {s:.3f}" for w, s in nearest(q, 5, ("king", "man", "woman"))))

print("\n④ 矢印は平行か？（差のベクトルどうしの cos）")
for (a, b), (c, d) in [(("king", "queen"), ("man", "woman")), (("king", "queen"), ("uncle", "aunt")), (("king", "queen"), ("cat", "car"))]:
    print(f"  cos({a}−{b}, {c}−{d}) = {cos(vec(a) - vec(b), vec(c) - vec(d)):.3f}")
print(f"  |king−queen| = {np.linalg.norm(vec('king') - vec('queen')):.2f}, |man−woman| = {np.linalg.norm(vec('man') - vec('woman')):.2f}")

print("\n⑤ ほかの足し算・引き算（入れた3語は除く、上位3つ）")
tests = [("paris", "france", "japan"), ("tokyo", "japan", "france"), ("walked", "walk", "swim"), ("bigger", "big", "small"),
         ("sushi", "japan", "italy"), ("strawberry", "red", "yellow"), ("brother", "man", "woman")]
for a, b, c in tests:
    r = nearest(vec(a) - vec(b) + vec(c), 3, (a, b, c))
    print(f"  {a} − {b} + {c} →", ", ".join(f"{w} {s:.3f}" for w, s in r))

print("\n⑥ 言えないこと用：社会の偏りも写る（Bolukbasi ほか 2016 と同じ問い）")
for a, b, c in [("doctor", "man", "woman"), ("programmer", "man", "woman")]:
    r = nearest(vec(a) - vec(b) + vec(c), 3, (a, b, c))
    print(f"  {a} − {b} + {c} →", ", ".join(f"{w} {s:.3f}" for w, s in r))

print("\n⑦ 正答率の目安：男女ペアどうしの問題（a − b + d → c になるか）で、1位が正解だった数")
pairs = [("king", "queen"), ("man", "woman"), ("boy", "girl"), ("brother", "sister"), ("father", "mother"), ("uncle", "aunt"), ("husband", "wife"), ("prince", "princess"), ("son", "daughter"), ("nephew", "niece")]
ok = tot = 0
for i, (a, b) in enumerate(pairs):
    for c, d in pairs[i + 1:i + 3]:
        r = nearest(vec(a) - vec(b) + vec(d), 1, (a, b, d))[0][0]
        ok += r == c; tot += 1
print(f"  {ok}/{tot} 問正解（入れた3語は除外）")


def rank(v, target, exclude=()):
    s = unit @ (v / np.linalg.norm(v))
    order = [words[i] for i in np.argsort(-s) if words[i] not in exclude]
    return order.index(target) + 1


print("\n⑧ 失敗の例：brother − man + woman で sister は何位？")
print(f"  sister は {rank(vec('brother') - vec('man') + vec('woman'), 'sister', ('brother', 'man', 'woman'))} 位")
print(f"  （king − man + woman で queen は {rank(q, 'queen', ('king', 'man', 'woman'))} 位、除外しないと {rank(q, 'queen')} 位）")
