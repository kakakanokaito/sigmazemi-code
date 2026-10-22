# 第1回「次の1文字を当てるだけ」の実験コード（標準ライブラリだけで動く）
# 使い方: python3 next_char.py wagahaiwa_nekodearu.txt
#   テキストは青空文庫『吾輩は猫である』(夏目漱石, 著作権切れ) のルビ付きテキストを想定。
#   Google Colab でもそのまま動く。
import math, random, re, sys
from collections import Counter, defaultdict

def load(path):
    raw = open(path, encoding="shift_jis", errors="ignore").read()
    body = raw.split("-------------------------------------------------------")[-1]
    body = body.split("底本：")[0]
    body = re.sub(r"《.*?》|［＃.*?］|｜", "", body)    # ルビ・注記を消す
    return re.sub(r"\s+", "", body)                     # 改行や空白も消す

def train(text, n):
    """直前 n-1 文字 → 次の1文字 の出現回数を数える"""
    table = defaultdict(Counter)
    for i in range(len(text) - n + 1):
        table[text[i:i + n - 1]][text[i + n - 1]] += 1
    return table

def prob(table, vocab, ctx, ch, alpha=0.0001):
    """出現回数を確率にする（見たことのない文字にも少しだけ確率を残す）"""
    c = table.get(ctx, Counter())
    return (c[ch] + alpha) / (sum(c.values()) + alpha * len(vocab))

def surprise(table, vocab, text, n):
    """1文字あたりの平均『びっくりポイント』= 交差エントロピー（単位: ビット）"""
    total = 0.0
    for i in range(n - 1, len(text)):
        total += -math.log2(prob(table, vocab, text[i - n + 1:i], text[i]))
    return total / (len(text) - n + 1)

def generate(table, start, n, length=60, seed=0):
    rng = random.Random(seed)
    out = start
    for _ in range(length):
        c = table.get(out[len(out) - (n - 1):] if n > 1 else "")
        if not c:
            break
        chars, counts = zip(*c.items())
        out += rng.choices(chars, weights=counts)[0]
    return out

if __name__ == "__main__":
    text = load(sys.argv[1])
    cut = int(len(text) * 0.9)
    train_text, test_text = text[:cut], text[cut:cut + 20000]  # 最後の1割は「テスト用」で学習に使わない
    vocab = set(text)
    print(f"文字数 {len(text):,} / 文字の種類 {len(vocab):,}")

    for n in [1, 2, 3, 4, 5]:
        t = train(train_text, n)
        print(f"\n=== 直前 {n-1} 文字を見るAI ===")
        print(f"びっくりポイント 学習に使った文章: {surprise(t, vocab, train_text[:20000], n):.2f} / "
              f"初めて見る文章: {surprise(t, vocab, test_text, n):.2f} ビット/文字")
        print("作った文章:", generate(t, "吾輩は猫で"[:max(n - 1, 0)] or "吾", n))

    # 「猫で」の次の1文字の確率トップ5（直前2文字を見るAIの中身）
    t = train(train_text, 3)
    c = t["猫で"]
    s = sum(c.values())
    print("\n「猫で」の次に来る文字（直前2文字を見るAIの中身）:")
    for ch, k in c.most_common(5):
        print(f"  {ch}: {k/s:.1%}  びっくりポイント {-math.log2(k/s):.2f}")
