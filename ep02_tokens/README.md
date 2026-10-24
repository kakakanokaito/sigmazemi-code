# 第2回：なぜAIは strawberry の r を数え間違えるのか（トークン）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kakakanokaito/sigmazemi-code/blob/main/ep02_tokens/02_tokens.ipynb)

動画：シグマゼミ 第2回（2026年10月24日公開）。上の Open In Colab から、ブラウザだけで全部動かせます

| ファイル | やること | 必要なもの |
|---|---|---|
| `tokens_demo.py` | OpenAI公式の区切り方（tiktoken、GPT-4o の o200k_base）で文を区切り、AIに届く番号を表示 | `pip install tiktoken` |
| `tokens_compare.py` | 同じ意味の英語と日本語を、GPT-4o と GPT-4 の区切り方で数えて比べる | `pip install tiktoken` |
| `bpe_demo.py` | 『吾輩は猫である』の最初の10万文字で、自分でBPE（いちばんよく隣り合う2つをくっつける）を300回 | Python だけ（第1回の `ep01_next_token/` を使う） |

```
pip install tiktoken
python3 tokens_demo.py
python3 tokens_compare.py
python3 bpe_demo.py ../ep01_next_token/wagahaiwa_nekodearu.txt
```

動画に出てくる数字（2026-10-23 に実行）：
- " strawberry"（文の中）＝ 1トークン、番号 101830。単独の "strawberry" は st | raw | berry の3つ
- 「I love eating strawberries very much.」37文字 → 7トークン
- 「私はイチゴを食べることがとても大好きです。」21文字 → GPT-4o で16トークン、GPT-4 で23トークン
- 自作BPE：1回目は「っ」+「て」→「って」（869回）、300回で 100,000 → 72,223 トークン

やってみよう：自分の名前や好きな文を tokens_demo.py に入れて、何トークンになるか数えてみてください。
