# bleed.py の確かめ。python3 test_bleed.py でも pytest でも動く。
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bleed import Lexicon, bleed, norm

LEX = Lexicon()
A, B, C = "方言A", "方言B", "方言C"

# (文, 方言, 漏れのはず, 当たるはずの言い回し or None)
CASES = [
    # 「〜」は何でもよい
    ("今日は晴れだネ😊", C, True, "〜だネ😊"),
    ("晴れだネ😊明日もね", C, True, "〜だネ😊"),
    ("今日は晴れだね。", C, False, None),
    ("いいゾこれ", A, True, "いいゾ〜これ"),
    ("いいゾ〜〜これ", A, True, "いいゾ〜これ"),
    # 括弧の注釈：全角でも半角でも
    ("答えは12です（意味深）", A, True, "〜（意味深）"),
    ("答えは12です(意味深)", A, True, "〜（意味深）"),
    ("答えは12です（確信）", A, True, "〜（確信）"),
    # 全角・半角の記号
    ("ファッ!?", A, True, "ファッ！？"),
    ("ありがとナス!", A, True, "ありがとナス！"),
    # 絵文字・顔文字
    ("コーヒーを飲もう☕", C, False, None),
    ("ゆっくり休むんだヨ🍵😊", C, True, "ゆっくり休むんだヨ🍵😊"),
    ("お疲れサマ☕✨", C, True, "お疲れサマ☕️✨"),   # ☕️ と ☕ は同じ
    ("それは大変(^_^;)", C, True, "(^_^;)"),
    ("それは大変（＾＿＾；）", C, True, "(^_^;)"),  # 全角の顔文字
    # 草は数えない
    ("それは草", B, False, None),
    ("負けてて草", B, False, None),
    ("大草原不可避", B, True, "大草原不可避"),
    # 読点が1文に4つ以上
    ("今日は、朝から、雨で、寒くて、大変だった。", C, True, "読点4つ以上"),
    ("今日は、朝から、雨で寒い。明日は、晴れ、らしい。", C, False, None),
    # たまたま入っているだけのもの
    ("ワインを飲む", B, False, None),
    ("ハワイに行く", B, False, None),
    ("ワイはそう思う", B, True, "ワイ"),
    ("リンゴを3個買った", B, False, None),
    ("負けたンゴ", B, True, "〜ンゴ"),
    ("一緒にやろう", B, False, None),
    ("答えは12やろ", B, True, "〜やろ"),
    ("明日は雨模様です", B, False, None),
    ("なお間に合わん模様", B, True, "なお間に合わん模様"),
    ("定期券を買う", B, False, None),
    ("ゾウは大きい", A, False, None),
    ("それでいいんだゾ", A, True, "〜だゾ"),
    ("133-45=88", B, False, None),
    ("33-4", B, True, "33-4"),
    ("すき焼きを作る", B, False, None),
    ("ラーメンすき", B, True, "〜すき"),
    ("いやな予感", B, False, None),
    ("せやな", B, True, "せやな"),
    ("有能な人材を探す", B, False, None),
    ("ニキビができた", B, False, None),
    ("ママ？", B, False, None),
    ("マ？", B, True, "マ？"),
    # ふつうの答え
    ("2+3の答えは5です。", A, False, None),
    ("2+3の答えは5です。", B, False, None),
    ("2+3の答えは5です。", C, False, None),
    ("「源氏物語」は紫式部が書いた。", C, False, None),   # かぎかっこは数えない
]


def test_cases():
    for text, d, want, phrase in CASES:
        got, hits = bleed(text, d, LEX)
        assert got == want, (text, d, hits)
        if phrase:
            assert phrase in hits, (text, d, phrase, hits)


def test_norm():
    assert norm("（意味深）") == "(意味深)"
    assert norm("☕️") == "☕"
    assert norm("ｗｗｗ") == "www"


def test_lexicon_loaded():
    assert set(LEX.dialects) >= {A, B, C}
    for d in (A, B, C):
        assert len(LEX.rules[d]) >= 50
    assert LEX.comma.get(C) == 4
    assert "草" not in [p for p, _ in LEX.rules[B]]


if __name__ == "__main__":
    test_cases(); test_norm(); test_lexicon_loaded()
    print(f"OK：{len(CASES)}例 ＋ 正規化3 ＋ 語録の読み込み")
