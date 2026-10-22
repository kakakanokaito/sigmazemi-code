#!/usr/bin/env python3
"""
シグマゼミ 第1回検証コード
「AIは次の言葉を当てるだけ？ — 吾輩は猫であるで自作言語モデルを作って検証してみた」

【概要】
夏目漱石の『吾輩は猫である』（青空文庫 約30万文字）を使い、
直前の N 文字の出現確率から「次の1文字」を予測・サンプリングする
自作マルコフ連鎖（N-gram言語モデル）を構築します。

【特徴】
- 外部ライブラリ不要（Python標準ライブラリのみで動作）
- 青空文庫からの自動ダウンロード＆前処理
- 再現性100%（乱数シード固定）
- 実験結果を JSON ファイル (experiment_results.json) に自動出力
"""

import os
import sys
import math
import json
import random
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict

# 青空文庫 URL
AOZORA_ZIP_URL = "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "789_ruby_5639.zip")
TXT_PATH = os.path.join(BASE_DIR, "wagahaiwa_nekodearu.txt")
RESULTS_JSON = os.path.join(BASE_DIR, "experiment_results.json")


def ensure_text_file():
    """テキストファイルがなければ青空文庫から自動取得して解凍"""
    if os.path.exists(TXT_PATH):
        return TXT_PATH

    print(f"[*] 青空文庫から『吾輩は猫である』をダウンロード中...\n    URL: {AOZORA_ZIP_URL}")
    req = urllib.request.Request(
        AOZORA_ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as resp, open(ZIP_PATH, "wb") as f:
        f.write(resp.read())

    print("[*] ZIPアーカイブを解凍中...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".txt"):
                with zf.open(name) as src, open(TXT_PATH, "wb") as dst:
                    dst.write(src.read())
                break

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    print(f"[+] 準備完了: {TXT_PATH}")
    return TXT_PATH


def load_and_clean_text(path):
    """青空文庫形式のShift_JISテキストを読み込み、ルビ・注記・空白を除去"""
    raw = open(path, encoding="shift_jis", errors="ignore").read()
    # ヘッダーとフッター（底本情報など）を除去
    body = raw.split("-------------------------------------------------------")[-1]
    body = body.split("底本：")[0]
    # ルビ（《...》）、字下げ注記（［＃...］）、区切り線（｜）を除去
    body = re.sub(r"《.*?》|［＃.*?］|｜", "", body)
    # 改行や空白を除去して1つの連続した文字ストリームにする
    clean = re.sub(r"\s+", "", body)
    return clean


def train_ngram(text, n):
    """直前 n-1 文字のコンテキストから次の1文字の出現頻度を記録"""
    table = defaultdict(Counter)
    for i in range(len(text) - n + 1):
        context = text[i:i + n - 1]
        next_ch = text[i + n - 1]
        table[context][next_ch] += 1
    return table


def get_prob(table, vocab_size, context, ch, alpha=0.0001):
    """ラプラス平滑化（未観測文字にも極小確率を残す）を適用した確率計算"""
    counter = table.get(context, Counter())
    total = sum(counter.values())
    return (counter[ch] + alpha) / (total + alpha * vocab_size)


def compute_surprise(table, vocab_size, text, n):
    """
    1文字あたりの平均『びっくりポイント』（交差エントロピー H = -log2 P(x)）
    単位: ビット/文字。値が小さいほど「AIにとって予想通りでびっくりしない」。
    """
    total_bits = 0.0
    valid_len = len(text) - n + 1
    if valid_len <= 0:
        return 0.0

    for i in range(n - 1, len(text)):
        ctx = text[i - n + 1:i]
        ch = text[i]
        p = get_prob(table, vocab_size, ctx, ch)
        total_bits += -math.log2(p)

    return total_bits / valid_len


def generate_text(table, start_text, n, length=65, seed=0):
    """自作AIによる文章自動生成（サイコロサンプリング）"""
    rng = random.Random(seed)
    out = start_text
    ctx_len = max(n - 1, 0)

    for _ in range(length):
        ctx = out[len(out) - ctx_len:] if ctx_len > 0 else ""
        counter = table.get(ctx)
        if not counter:
            break
        chars, counts = zip(*counter.items())
        chosen = rng.choices(chars, weights=counts)[0]
        out += chosen

    return out


def main():
    print("=" * 60)
    print("  シグマゼミ【自作AI実験】言語モデルは次の1文字を当てるだけ？")
    print("=" * 60)

    # 1. テキスト取得と前処理
    txt_file = ensure_text_file()
    text = load_and_clean_text(txt_file)
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)

    # 90%を学習用、10%（最大20,000文字）を未知テスト用として分割
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    test_text = text[split_idx:split_idx + 20000]

    print(f"総文字数: {len(text):,} 文字")
    print(f"語彙数（ユニーク文字種）: {vocab_size:,} 種類")
    print(f"学習データ: {len(train_text):,} 文字 / テストデータ: {len(test_text):,} 文字")
    print("-" * 60)

    results = {
        "metadata": {
            "source": "夏目漱石『吾輩は猫である』（青空文庫）",
            "total_chars": len(text),
            "vocab_size": vocab_size,
            "train_chars": len(train_text),
            "test_chars": len(test_text)
        },
        "models": [],
        "nekode_distribution": []
    }

    # 2. 直前 0〜4 文字を見るAI（N=1〜5）の学習と検証
    for n in [1, 2, 3, 4, 5]:
        ctx_chars = n - 1
        t = train_ngram(train_text, n)

        # びっくりポイント（交差エントロピー）の算出
        train_eval = train_text[:20000]
        train_surprise = compute_surprise(t, vocab_size, train_eval, n)
        test_surprise = compute_surprise(t, vocab_size, test_text, n)

        # 文章生成
        prompt = "吾輩は猫で"[:ctx_chars] if ctx_chars > 0 else "吾"
        gen = generate_text(t, prompt, n, length=60, seed=0)

        model_info = {
            "n": n,
            "context_chars": ctx_chars,
            "label": f"直前 {ctx_chars} 文字を見るAI",
            "train_surprise_bits": round(train_surprise, 2),
            "test_surprise_bits": round(test_surprise, 2),
            "generated_text": gen
        }
        results["models"].append(model_info)

        print(f"\n【直前 {ctx_chars} 文字を見るAI】(N={n})")
        print(f"  ・学習データのびっくり度 : {train_surprise:.2f} ビット/文字")
        print(f"  ・初めて見る文章のびっくり度: {test_surprise:.2f} ビット/文字")
        print(f"  ・生成された文章: {gen}")

    # 3. 「猫で」の次に来る文字の確率サイコロ（N=3: 直前2文字を見るAI）
    t3 = train_ngram(train_text, 3)
    c = t3["猫で"]
    total_nekode = sum(c.values())

    print("\n" + "=" * 60)
    print("【実証】AIが『猫で』の次に振るサイコロの確率（上位5文字）:")
    print("=" * 60)

    for ch, count in c.most_common(5):
        prob = count / total_nekode
        surprise_val = -math.log2(prob)
        item = {
            "char": ch,
            "count": count,
            "probability_pct": round(prob * 100, 1),
            "surprise_bits": round(surprise_val, 2)
        }
        results["nekode_distribution"].append(item)
        print(f"  「{ch}」: {prob * 100:5.1f}%  (出現 {count} 回, びっくり度 {surprise_val:.2f} ビット)")

    # 4. JSONに出力
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n[+] 全実験結果を保存しました:", RESULTS_JSON)
    print("=" * 60)


if __name__ == "__main__":
    main()
