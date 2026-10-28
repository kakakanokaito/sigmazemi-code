# 第4回：文字しか読んでいないAIの頭の中に、日本地図はあるのか

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kakakanokaito/sigmazemi-code/blob/main/ep04_japan_map/04_japan_map.ipynb)

日本の都市（人口1.5万人以上、約1,250個）の名前だけをオープンなAI（Qwen2.5）に入れ、そのときの内部の数字から緯度・経度を直線の式（リッジ回帰）で当てます。答えを見せていない都市だけで当てて、ずれを km で測ります。

- 元の研究：Wes Gurnee, Max Tegmark「Language Models Represent Space and Time」ICLR 2024（arXiv:2310.02207）、コード https://github.com/wesg52/world-models
- 都市のデータ：GeoNames cities15000（CC BY 4.0、https://www.geonames.org/）
- くらべる相手：①全部の都市を「日本のまん中」と答えた場合 ②答えをシャッフルして同じことをした場合 ③都道府県ごと隠して当てた場合（きびしい版）

```
python3 japan_map.py --model Qwen/Qwen2.5-7B            # 英語の都市名
python3 japan_map.py --model Qwen/Qwen2.5-7B --lang ja  # 漢字の都市名
python3 japan_map.py --model Qwen/Qwen2.5-32B --load_4bit  # 32B・72B は4ビットで
```

注意（言えないこと）：内部の数字から位置を当てられても、AIが答えるときにその「地図」を使っている証拠にはなりません（元の論文も同じことを書いています）。一番よい層は結果を見て選んでいるので、少しだけ甘めの数字です。

試走（2026-10-25、CPU、`trial_cpu/`）：0.5B（1,247都市）は当たらず、全部まん中と答えるのと同じ（ずれの中央値 274km、まん中 268km）。1.5B（大きい300都市）では R² 緯度0.46・経度0.38、ずれの中央値 186km（まん中 231km）、東京 141km、鹿児島 358km。

本番（2026-10-25、Colab A100 80GB）：大きい50都市のずれの中央値 0.5B 245km → 1.5B 158km → 3B 140km → 7B 109km → 14B 94km → 32B 94km → 72B 80km。表は `results_colab/まとめ.md`。
