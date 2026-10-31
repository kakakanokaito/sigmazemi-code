# 第9回：数学の答えの採点。答えの文から数（整数・小数・分数・％）を抜き出し、正解とくらべる。
# 採点は2通り：
#   ゆるい  … 答えに出た数のどれか1つでも正解と一致すれば正答（途中の式に正解が出ていても正答）
#   きびしい … 答えの「最後に出た数」だけで判定（正解が2つある問題は最後の2つ）
# 正解の書き方（data/eval_questions.csv の列「正解」）：
#   600 / 0.6 / 1/2 … 数。1/2 と 0.5 と 50% と 3/6 は同じ値として扱う。全角数字・カンマ（1,024）・単位（円・個・km・cm³）は気にしない
#   2;3             … 正解が2つ（両方とも出たら正答）
#   5√2             … 数でない答えは、空白や LaTeX の書き方をそろえて「文字が含まれるか」で判定（ゆるい・きびしいは同じ結果）
#   a|b             … どちらでもよい（書き方の違い）
#   小数は、小数第2位以上まで書いた答えなら四捨五入で一致すれば正答（1/6 に 0.17 は正答、0.2 は誤り）
#   符号（−）は見ない（今回の30問に負の答えはない）。漢数字（五百）は数えない
# 使い方：python3 score_math.py --answers <answers_*.csv のフォルダ> --out <math_accuracy.csv>
#         （そのフォルダの答えを採点し直して、方言×教えた数の正答率を書く）
import argparse, csv, glob, os, re, unicodedata
from fractions import Fraction

SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹"
NUM = r"\d+(?:\.\d+)?"
TOKEN = re.compile(rf"({NUM})\s*/\s*({NUM})|({NUM})\s*%|({NUM})")


def _clean(text):
    """数を抜き出す前の下ごしらえ：LaTeX・全角・指数・添字・桁区切りをそろえる。"""
    t = text or ""
    t = re.sub(f"[{SUP}]+", "", t)                         # x² cm³ の小さい数は数に入れない（NFKC の前に消す）
    t = unicodedata.normalize("NFKC", t).replace("⁄", "/").replace("∕", "/")
    t = re.sub(r"\\[dt]?frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", r"(\1)/(\2)", t)   # \frac{1}{2} → (1)/(2)
    t = re.sub(r"\((\d+(?:\.\d+)?)\)/\((\d+(?:\.\d+)?)\)", r"\1/\2", t)
    t = re.sub(r"(\d+)\s*分の\s*(\d+)", r"\2/\1", t)        # 2分の1 → 1/2
    t = t.replace("\\%", "%")
    t = re.sub(r"\^\s*\{[^{}]*\}|\^\s*\d+", "", t)          # 指数 x^2・2^{10} は数に入れない
    t = re.sub(r"_\s*\{[^{}]*\}|_\s*\d+", "", t)            # 添字 a_{10}・a_1 も
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)          # 1,024 → 1024（2,3 はそのまま）
    return t


def extract_numbers(text):
    """答えの文に出てくる数を、出てきた順に [(値 Fraction, 小数の桁数)] で返す。"""
    out = []
    for m in TOKEN.finditer(_clean(text)):
        if m.group(1) is not None:
            a, b = Fraction(m.group(1)), Fraction(m.group(2))
            if b == 0: continue
            out.append((a / b, None))
        elif m.group(3) is not None:
            s = m.group(3)
            places = (len(s.split(".")[1]) if "." in s else 0) + 2
            out.append((Fraction(s) / 100, places))
        else:
            s = m.group(4)
            out.append((Fraction(s), len(s.split(".")[1]) if "." in s else 0))
    return out


def _as_number(s):
    s = unicodedata.normalize("NFKC", s).strip().replace(",", "")
    m = re.fullmatch(rf"({NUM})\s*/\s*({NUM})|({NUM})\s*%|({NUM})", s)
    if not m: return None
    if m.group(1): return Fraction(m.group(1)) / Fraction(m.group(2))
    if m.group(3): return Fraction(m.group(3)) / 100
    return Fraction(m.group(4))


def _same(want, got):
    v, places = got
    if v == want: return True
    if places is not None and places >= 2:    # 小数第2位以上まで書いてあれば、四捨五入で一致を許す
        q = Fraction(1, 10 ** places)
        return abs(want - v) <= q / 2
    return False


def _norm_text(s):
    t = (s or "").replace("\\sqrt", "√")
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\\(cdot|times|left|right)|[×*・\s^{}]", "", t)
    return t.lower()


def parse_answer(spec):
    """正解の書き方 → [書き方の候補]。候補は [(種類, 値)] で、種類は "num" か "text"。"""
    alts = []
    for alt in (spec or "").split("|"):
        items = [x.strip() for x in alt.split(";") if x.strip()]
        if not items: continue
        alts.append([("num", _as_number(x)) if _as_number(x) is not None else ("text", _norm_text(x)) for x in items])
    return alts


def score(answer, spec):
    """答えの文と正解の書き方 → (ゆるい, きびしい) の正答かどうか。正解が空なら (None, None)。"""
    alts = parse_answer(spec)
    if not alts: return None, None
    nums = extract_numbers(answer)
    text = _norm_text(answer)
    loose = strict = False
    for items in alts:
        ok_l = all((any(_same(v, g) for g in nums) if k == "num" else v in text) for k, v in items)
        want = [v for k, v in items if k == "num"]
        if want:
            tail = list(nums[-len(want):]) if len(nums) >= len(want) else None
            ok_s = tail is not None and all(v in text for k, v in items if k == "text")
            if ok_s:   # 最後の k 個が、正解の k 個と（順番は問わず）1対1で一致
                rest = list(tail)
                for w in want:
                    hit = next((i for i, g in enumerate(rest) if _same(w, g)), None)
                    if hit is None: ok_s = False; break
                    rest.pop(hit)
        else:
            ok_s = ok_l
        loose |= ok_l; strict |= ok_s
    return loose, strict


# ---------- 答えのファイルをまとめて採点 ----------
def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]


def accuracy_rows(dialect, size, rows, keys):
    """1つの answers_*.csv（rows）を採点して math_accuracy.csv の1行を返す。keys は {id: 正解}。"""
    L = S = n = 0
    for r in rows:
        spec = keys.get(r["id"], "")
        if not spec: continue
        l, s = score(r["答え"], spec)
        L += l; S += s; n += 1
    if n == 0: return None
    return {"方言": dialect, "size": size, "正答率_ゆるい": round(L / n, 4), "正答率_きびしい": round(S / n, 4), "件数": n}


ACC_COLS = ["方言", "size", "正答率_ゆるい", "正答率_きびしい", "件数"]


def rescore_dir(folder, questions):
    keys = {q["id"]: q.get("正解", "") for q in read_csv(questions) if q.get("分野") == "数学"}
    out = []
    for p in glob.glob(os.path.join(folder, "answers_*.csv")):
        m = re.fullmatch(r"answers_(.+)_(\d+)\.csv", os.path.basename(p))
        if not m: continue
        r = accuracy_rows(m.group(1), int(m.group(2)), read_csv(p), keys)
        if r: out.append(r)
    return sorted(out, key=lambda r: (r["方言"], r["size"]))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="answers_<方言>_<数>.csv があるフォルダ")
    ap.add_argument("--questions", default=os.path.join(here, "data", "eval_questions.csv"))
    ap.add_argument("--out", default="", help="既定は <フォルダ>/math_accuracy.csv")
    a = ap.parse_args()
    rows = rescore_dir(a.answers, a.questions)
    out = a.out or os.path.join(a.answers, "math_accuracy.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ACC_COLS); w.writeheader(); w.writerows(rows)
    print("数学の正答率（ゆるい／きびしい）")
    for r in rows:
        print(f"  {r['方言']} {r['size']:>4}組  {r['正答率_ゆるい']:.0%} ／ {r['正答率_きびしい']:.0%}  （{r['件数']}問）")
    print("保存：", out)
