# score_math.py の確かめ。python3 test_score_math.py でも pytest でも動く。
import os, sys
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_math import extract_numbers, score

# (答えの文, 正解, ゆるい, きびしい)
CASES = [
    ("120円 × 5 = 600円です。", "600", True, True),
    ("５個で６００円になります。", "600", True, True),            # 全角
    ("実質84円だゾ。", "600", False, False),
    ("600円…と思ったけど、実は84円だゾ。", "600", True, False),   # ゆるい と きびしい の違い
    ("合計は1,024です。", "1024", True, True),                   # 桁区切り
    ("確率は 50% です。", "1/2", True, True),
    ("確率は 0.5 です。", "1/2", True, True),
    ("\\frac{3}{6} = \\frac{1}{2}", "1/2", True, True),        # LaTeX の分数、3/6 も同じ値
    ("2分の1です", "1/2", True, True),
    ("答えは2/3だよ。", "1/2", False, False),
    ("約0.17です", "1/6", True, True),                          # 小数第2位まで → 四捨五入で一致
    ("約0.2です", "1/6", False, False),                         # 第1位までは許さない
    ("28.26 cm² です", "28.26", True, True),                    # cm² の 2 は数に入れない
    ("A = 3.14 \\times (3)^2 = 28.26 平方センチ", "28.26", True, True),
    ("x = 2, 3", "2;3", True, True),
    ("解は x=3 と x=2", "2;3", True, True),                      # 順番は問わない
    ("x = 2 です", "2;3", False, False),
    ("傾きは 2 、切片は 1 です。", "2;1", True, True),
    ("√50 = √(25×2) = 5√2", "5√2", True, True),
    ("5\\sqrt{2} になります", "5√2", True, True),
    ("(a+b)^2 = a^2 + 2ab + b^2", "a²+2ab+b²|a²+b²+2ab", True, True),
    ("a² + b² + 2ab です", "a²+2ab+b²|a²+b²+2ab", True, True),
    ("a² + b² です", "a²+2ab+b²|a²+b²+2ab", False, False),
    ("a_{10} = 2 + 9 \\times 3 = 29", "29", True, True),          # 添字 10 は数に入れない
    ("答えは29。サイコロを1回", "29", True, False),                # 途中で切れた答え
    ("数はありません", "5", False, False),
]


def test_cases():
    for text, spec, l, s in CASES:
        assert score(text, spec) == (l, s), (text, spec, score(text, spec), extract_numbers(text))


def test_extract():
    assert [v for v, _ in extract_numbers("１２０円×５＝６００円")] == [120, 5, 600]
    assert [v for v, _ in extract_numbers("3/4 + 1/6 = 11/12")] == [Fraction(3, 4), Fraction(1, 6), Fraction(11, 12)]
    assert [v for v, _ in extract_numbers("2時間30分")] == [2, 30]
    assert score("なんでも", "") == (None, None)


if __name__ == "__main__":
    test_cases(); test_extract()
    print(f"OK：{len(CASES)}例 ＋ 数の抜き出し4")
