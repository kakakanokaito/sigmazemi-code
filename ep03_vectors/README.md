# 第3回：なぜ「王 − 男 ＋ 女 ≒ 女王」になるのか（言葉のベクトル）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kakakanokaito/sigmazemi-code/blob/main/ep03_vectors/03_word_vectors.ipynb)

動画：シグマゼミ 第3回（2026年10月25日公開）。上の Open In Colab から、ブラウザだけで全部動かせます。

| ファイル | やること | 必要なもの |
|---|---|---|
| `vectors_demo.py` | GloVe（40万語×100次元）で cos を計算、王−男＋女、ほかの足し算、失敗例、偏り、男女ペア17問の正答数 | numpy、GloVe 100次元 |
| `cooccur_ratio.py` | text8（英語版ウィキペディアの最初の約1,700万語）で、king・queen・man・woman の前後5語を数えて「共起の比」を出す | Python だけ、text8 |
| `more_probes.py` | kitten の近所、矢印の長さ、cos の手計算、天才−かしこい＋なまけもの など | numpy、GloVe 100次元 |

```
pip install numpy
wget https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
wget https://github.com/RaRe-Technologies/gensim-data/releases/download/text8/text8.gz
python3 vectors_demo.py glove-wiki-gigaword-100.gz
python3 cooccur_ratio.py text8.gz
python3 more_probes.py glove-wiki-gigaword-100.gz
```

今日の式：cos θ = a・b / (|a| |b|)（手で1回：a=(3,4)、b=(4,3) → 24÷25 = 0.96）

動画に出てくる数字（2026-10-24 に実行）：
- cos：cat–dog 0.880、king–queen 0.751、strawberry–blueberry 0.576、cat–kitten 0.558、cat–car 0.311、strawberry–democracy 0.048
- king − man + woman → queen 0.783（入れた3語を除く）。除かないと1位は king 0.855
- paris − france + japan → tokyo 0.892、walked − walk + swim → swam 0.752、monday − work + play → saturday 0.816
- genius − smart + lazy → scoundrel 0.520、idiot 0.497
- brother − man + woman → daughter 0.895（sister は6位）。男女ペア17問で、1位が正解だったのは15問
- doctor − man + woman → nurse 0.776（入れた3語を除く）／除かないと doctor 0.866
- 共起の比（text8）：she は「king÷queen」0.16、「man÷woman」0.12。her は 0.12 と 0.10
- kitten の近所：1位 puppy 0.691

注意：text8 は GloVe の学習データ（Wikipedia＋Gigaword、60億語）とは別の、小さな文章で数えた例です。考え方は GloVe の論文（Pennington ほか 2014）の表1と同じです。

出典：Pennington, Socher, Manning (2014) GloVe／Mikolov, Yih, Zweig (2013)／Bolukbasi ほか (2016)／Nissim ほか (2020)。単語ベクトルと text8 は gensim-data から。

やってみよう：Colab の ④ で、好きな3語を入れて足し算・引き算してみてください。
