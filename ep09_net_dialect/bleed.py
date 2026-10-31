# 第9回「AIにネットの方言を教えたら」の漏れ判定（シグマゼミ）
# 答えの文に、その方言の言い回し（data/lexicon.csv）が1つでも入っていれば「漏れ」。
#   ・「〜」は「何が入ってもよい（同じ文の中で30字まで、空でもよい）」として扱う。
#     例：「〜だネ😊」は「だネ😊」が文中か文末にあれば一致。「いいゾ〜これ」は「いいゾこれ」「いいゾ〜〜これ」も一致。
#   ・全角と半角は同じとみなす（（意味深）と(意味深)、！と!、ｗとw）。絵文字の見た目の違い（☕️ と ☕）も同じとみなす。
#   ・数え方「数えない」の語（草 など、ネット全般で使う語）は数えない。
#   ・数え方「読点4」：1つの文に読点（、）が4つ以上あれば漏れ。
#   ・ほかの言葉の一部にたまたま入っているだけのもの（ワイン の「ワイ」、リンゴ の「ンゴ」、やろう の「やろ」、
#     雨模様 の「模様」、定期券 の「定期」など）は数えない。下の GUARD に一覧。
import csv, os, re, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
LEXICON = os.path.join(HERE, "data", "lexicon.csv")
WILD = "[^。!?\n]{0,30}?"          # 「〜」＝同じ文の中の0〜30字
KATA = "ァ-ヺー"                     # カタカナ（のばす棒も）
KANA_KANJI_AFTER = "ぃ-ゔァ-ヺ一-龯々"  # 後ろに続くと別の言葉になる文字（ぁ・あ＝のばしは許す）
SENT = re.compile(r"[^。!?\n]+")      # 文の区切り（正規化のあとの記号）


def norm(s):
    """全角・半角をそろえ、絵文字の見た目の違い（U+FE0F）を消す。〜 の仲間は 〜 にそろえる。"""
    s = unicodedata.normalize("NFKC", str(s)).replace("️", "")
    return re.sub("[~∼〰]", "〜", s)


# 言い回しごとの追加の条件（前に来てはいけない文字・後に来てはいけない文字）。キーは正規化前の言い回し。
_END = f"(?![{KANA_KANJI_AFTER}])"   # 文末のことば（後ろに字が続かない）
GUARD = {
    "ワイ": (f"(?<![{KATA}])", f"(?![{KATA}])"),            # ワイン・ハワイ・ワイファイ
    "ちなワイ": ("", f"(?![{KATA}])"),
    "ニキ": (f"(?<![{KATA}])", f"(?![{KATA}])"),            # ニキビ
    "ネキ": (f"(?<![{KATA}])", f"(?![{KATA}])"),
    "〜ゾ": (f"(?<![{KATA}])", f"(?![{KATA}])"),            # ゾウ・アマゾン・ゾーン
    "〜だゾ": ("", f"(?![{KATA}])"),
    "〜ンゴ": (f"(?<![{KATA}])", f"(?![{KATA}])"),          # リンゴ・マンゴー・タンゴ
    "マ？": (f"(?<![{KATA}])", ""),                          # ママ？・テーマ？
    "〜やで": ("", _END), "〜やね": ("", _END), "〜やろ": ("", _END),   # やろう・やね（屋根）
    "〜やな": ("", _END), "〜やぞ": ("", _END), "〜やん": ("", _END),   # いやな人・やんちゃ
    "〜すき": ("", _END),                                    # すき焼き・すきま
    "〜定期": ("", _END),                                    # 定期券・定期的
    "〜模様": ("(?<![雨空雪荒花柄縞玉様])", _END),          # 雨模様・花模様
    "〜の鑑": ("", "(?![定賞別])"),                          # 〜の鑑定
    "人間の鑑": ("", "(?![定賞別])"),
    "33-4": ("(?<![0-9])", "(?![0-9])"),                     # 133-45 などの計算
    "有能": ("", "(?!な|に|さ|で|だ|性|者)"),                # 有能な人材（ふつうの日本語）。言い切りだけ数える
}


def compile_phrase(p):
    lead, tail = GUARD.get(p, ("", ""))
    q = norm(p).strip()
    core = q.strip("〜")                     # 先頭・末尾の 〜 は「何でもよい」なので外す
    parts = [re.escape(x) for x in core.split("〜")]
    return re.compile(lead + WILD.join(parts) + tail)


class Lexicon:
    def __init__(self, path=LEXICON):
        self.rules = {}   # 方言 → [(言い回し, 正規表現)]
        self.comma = {}   # 方言 → 読点の数のしきい値
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                d, p, how = r["方言"].strip(), r["言い回し"].strip(), r.get("数え方", "語").strip() or "語"
                self.rules.setdefault(d, [])
                if how == "数えない":
                    continue
                if how.startswith("読点"):
                    self.comma[d] = int(how[2:] or 4)
                    continue
                self.rules[d].append((p, compile_phrase(p)))

    @property
    def dialects(self):
        return list(self.rules)

    def hits(self, text, dialect):
        """当たった言い回しの一覧（なければ空）。"""
        t = norm(text)
        out = [p for p, rx in self.rules[dialect] if rx.search(t)]
        k = self.comma.get(dialect)
        if k and any(s.count("、") + s.count(",") >= k for s in SENT.findall(t)):
            out.append(f"読点{k}つ以上")
        return out

    def bleed(self, text, dialect):
        """(漏れたか, 当たった言い回し)"""
        h = self.hits(text, dialect)
        return bool(h), h


_default = None


def bleed(text, dialect, lexicon=None):
    """関数1つで使う版：bleed("今日は晴れだネ😊", "方言C") → (True, ["〜だネ😊"])"""
    global _default
    if lexicon is None:
        if _default is None:
            _default = Lexicon()
        lexicon = _default
    return lexicon.bleed(text, dialect)
