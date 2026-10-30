# 第7回 追加実験（シグマゼミ、2026-10-30）
# ① 頭の中では分かっていた？：実在する町・作家・定理と、架空のものを同じ形の質問にして読ませ、答える直前の
#    頭の中の数字（隠れ状態）から「実在するか」を直線の式（ロジスティック回帰）で当てる。層ごとの正答率と、
#    「口では作り話をした質問」のうち何%を頭の中では「存在しない」側に置いていたかを出す（第5回と同じ取り出し方）。
# ② 10回聞いたら：同じ質問を10回（温度0.7）答えさせ、いちばん多い答えが何回出たか（そろい度）を数える。
#    本当のことはそろい、作り話はばらける、なら「何回か聞いてそろうか見る」で作り話を見分けられる。
# 元の研究：①Marks & Tegmark 2023（arXiv:2310.06824）の手口を「実在するか」に当てる。
#           ②SelfCheckGPT（Manakul ら 2023、arXiv:2303.08896）と同じ考え方【要確認：番号】。
# 使い方：python3 knows_and_repeat.py --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct
#         python3 knows_and_repeat.py --trial   （0.5B・CPU・各20問）
import argparse, json, os, re
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from judge import judge

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct")
ap.add_argument("--trial", action="store_true")
ap.add_argument("--n_rep", type=int, default=10, help="② で同じ質問を聞く回数")
ap.add_argument("--n_rep_q", type=int, default=40, help="② で使う質問の数（種類ごと・実在と架空それぞれ）")
ap.add_argument("--out", default="results")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
if args.trial:
    args.models, args.n_rep_q = "Qwen/Qwen2.5-0.5B-Instruct", 5
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(args.seed)

# ---------- 実在するもの（同じ形の質問にする）----------
REAL_TOWNS = [("北海道", t) for t in ["ニセコ", "美瑛", "東川", "余市", "白老", "洞爺湖", "中標津", "厚岸", "上士幌", "鹿追", "浦河"]] + [
    ("青森県", "大鰐"), ("青森県", "鰺ヶ沢"), ("青森県", "野辺地"), ("岩手県", "雫石"), ("岩手県", "紫波"), ("岩手県", "平泉"),
    ("岩手県", "大槌"), ("宮城県", "松島"), ("宮城県", "女川"), ("宮城県", "南三陸"), ("宮城県", "蔵王"), ("秋田県", "小坂"),
    ("秋田県", "八郎潟"), ("秋田県", "美郷"), ("山形県", "大江"), ("山形県", "庄内"), ("福島県", "猪苗代"), ("福島県", "只見"),
    ("福島県", "三春"), ("茨城県", "大洗"), ("茨城県", "境"), ("栃木県", "益子"), ("栃木県", "那須"), ("栃木県", "茂木"),
    ("群馬県", "草津"), ("群馬県", "みなかみ"), ("埼玉県", "小川"), ("埼玉県", "寄居"), ("埼玉県", "長瀞"), ("千葉県", "九十九里"),
    ("千葉県", "鋸南"), ("千葉県", "酒々井"), ("神奈川県", "箱根"), ("神奈川県", "葉山"), ("神奈川県", "真鶴"), ("神奈川県", "湯河原"),
    ("神奈川県", "大磯"), ("新潟県", "湯沢"), ("新潟県", "出雲崎"), ("富山県", "立山"), ("富山県", "入善"), ("石川県", "能登"),
    ("石川県", "穴水"), ("福井県", "永平寺"), ("福井県", "美浜"), ("山梨県", "富士河口湖"), ("山梨県", "身延"), ("長野県", "軽井沢"),
    ("長野県", "小布施"), ("長野県", "木曽"), ("岐阜県", "揖斐川"), ("静岡県", "西伊豆"), ("静岡県", "南伊豆"), ("静岡県", "東伊豆"),
    ("愛知県", "東浦"), ("愛知県", "武豊"), ("三重県", "大台"), ("三重県", "紀北"), ("滋賀県", "竜王"), ("京都府", "与謝野"),
    ("京都府", "伊根"), ("大阪府", "島本"), ("大阪府", "能勢"), ("兵庫県", "香美"), ("兵庫県", "佐用"), ("奈良県", "吉野"),
    ("奈良県", "斑鳩"), ("和歌山県", "白浜"), ("和歌山県", "那智勝浦"), ("和歌山県", "串本"), ("和歌山県", "高野"), ("鳥取県", "大山"),
    ("鳥取県", "湯梨浜"), ("島根県", "隠岐の島"), ("島根県", "奥出雲"), ("島根県", "津和野"), ("岡山県", "奈義"), ("岡山県", "和気"),
    ("広島県", "熊野"), ("山口県", "周防大島"), ("徳島県", "上勝"), ("徳島県", "美波"), ("香川県", "小豆島"), ("香川県", "琴平"),
    ("香川県", "直島"), ("愛媛県", "内子"), ("愛媛県", "砥部"), ("高知県", "四万十"), ("高知県", "梼原"), ("福岡県", "篠栗"),
    ("佐賀県", "有田"), ("佐賀県", "太良"), ("長崎県", "波佐見"), ("長崎県", "小値賀"), ("熊本県", "南小国"), ("熊本県", "高森"),
    ("大分県", "九重"), ("大分県", "玖珠"), ("宮崎県", "高千穂"), ("宮崎県", "綾"), ("鹿児島県", "屋久島"), ("鹿児島県", "湧水"),
    ("鹿児島県", "長島"), ("沖縄県", "本部"), ("沖縄県", "北谷")]
REAL_AUTHORS = ["夏目漱石", "芥川龍之介", "太宰治", "宮沢賢治", "川端康成", "三島由紀夫", "森鴎外", "樋口一葉", "村上春樹", "江戸川乱歩",
                "島崎藤村", "谷崎潤一郎", "宮部みゆき", "東野圭吾", "司馬遼太郎", "恩田陸", "湊かなえ", "伊坂幸太郎", "重松清", "吉本ばなな",
                "宮本輝", "遠藤周作", "安部公房", "井上靖", "角田光代", "辻村深月", "森見登美彦", "朝井リョウ", "原田マハ", "小川洋子",
                "山田詠美", "筒井康隆", "星新一", "赤川次郎", "西村京太郎", "松本清張", "横溝正史", "池井戸潤", "綿矢りさ", "金原ひとみ",
                "又吉直樹", "村田沙耶香", "川上弘美", "江國香織", "桐野夏生", "京極夏彦", "浅田次郎", "北方謙三", "大江健三郎", "井伏鱒二",
                "中島敦", "梶井基次郎", "坂口安吾", "志賀直哉", "武者小路実篤", "有島武郎", "泉鏡花", "幸田露伴", "尾崎紅葉", "有川浩"]
REAL_THEOREMS = ["ハイネ＝ボレルの定理", "ボルツァーノ＝ワイエルシュトラスの定理", "ケーリー＝ハミルトンの定理", "ピタゴラスの定理",
                 "フェルマーの小定理", "フェルマーの最終定理", "ガウス＝ボネの定理", "ストークスの定理", "グリーンの定理", "ロルの定理",
                 "テイラーの定理", "ラグランジュの定理", "シローの定理", "ハーン＝バナッハの定理", "ゲーデルの不完全性定理", "チェバの定理",
                 "メネラウスの定理", "ウィルソンの定理", "ディリクレの算術級数定理", "ブラウワーの不動点定理", "ジョルダンの曲線定理",
                 "リーマン＝ロッホの定理", "アティヤ＝シンガーの指数定理", "ラドン＝ニコディムの定理", "フビニの定理", "ストーン＝ワイエルシュトラスの定理",
                 "アルツェラ＝アスコリの定理", "ベールのカテゴリー定理", "リウヴィルの定理", "ピカールの定理", "ド・モアブルの定理",
                 "ラムゼーの定理", "ホールの結婚定理", "メンガーの定理", "クラトフスキーの定理", "ネーターの定理", "ピックの定理",
                 "コーシーの積分定理", "オイラーの多面体定理", "バナッハ＝タルスキーの定理"]

# ---------- 架空のもの（実在と同じ数）----------
# 見た目で見分けられないように、実在の名前の部品を組み替えて作る（「見慣れない字かどうか」で当てられないようにする）
#   町：実在の町の名前の前半と後半をつなぐ／作家：実在の作家の姓と名を入れ替える／定理：実在の定理の人名を入れ替える
#   組み替えた結果が実在の名前と同じになったものは除く。偶然どこかに実在する可能性は残る【要確認：上位の例は目で見る】
PREFS = sorted({p for p, _ in REAL_TOWNS})
AUTH_SPLIT = {"夏目漱石": ("夏目", "漱石"), "芥川龍之介": ("芥川", "龍之介"), "太宰治": ("太宰", "治"), "宮沢賢治": ("宮沢", "賢治"),
              "川端康成": ("川端", "康成"), "三島由紀夫": ("三島", "由紀夫"), "森鴎外": ("森", "鴎外"), "樋口一葉": ("樋口", "一葉"),
              "村上春樹": ("村上", "春樹"), "島崎藤村": ("島崎", "藤村"), "谷崎潤一郎": ("谷崎", "潤一郎"), "宮部みゆき": ("宮部", "みゆき"),
              "東野圭吾": ("東野", "圭吾"), "司馬遼太郎": ("司馬", "遼太郎"), "恩田陸": ("恩田", "陸"), "湊かなえ": ("湊", "かなえ"),
              "伊坂幸太郎": ("伊坂", "幸太郎"), "重松清": ("重松", "清"), "宮本輝": ("宮本", "輝"), "遠藤周作": ("遠藤", "周作"),
              "安部公房": ("安部", "公房"), "井上靖": ("井上", "靖"), "角田光代": ("角田", "光代"), "辻村深月": ("辻村", "深月"),
              "森見登美彦": ("森見", "登美彦"), "朝井リョウ": ("朝井", "リョウ"), "原田マハ": ("原田", "マハ"), "小川洋子": ("小川", "洋子"),
              "山田詠美": ("山田", "詠美"), "筒井康隆": ("筒井", "康隆"), "星新一": ("星", "新一"), "赤川次郎": ("赤川", "次郎"),
              "松本清張": ("松本", "清張"), "横溝正史": ("横溝", "正史"), "池井戸潤": ("池井戸", "潤"), "綿矢りさ": ("綿矢", "りさ"),
              "川上弘美": ("川上", "弘美"), "江國香織": ("江國", "香織"), "桐野夏生": ("桐野", "夏生"), "京極夏彦": ("京極", "夏彦"),
              "浅田次郎": ("浅田", "次郎"), "北方謙三": ("北方", "謙三"), "井伏鱒二": ("井伏", "鱒二"), "中島敦": ("中島", "敦"),
              "坂口安吾": ("坂口", "安吾"), "志賀直哉": ("志賀", "直哉"), "泉鏡花": ("泉", "鏡花"), "幸田露伴": ("幸田", "露伴"),
              "尾崎紅葉": ("尾崎", "紅葉"), "有川浩": ("有川", "浩")}
THEO_NAMES = ["ハイネ", "ボレル", "ボルツァーノ", "ワイエルシュトラス", "ケーリー", "ハミルトン", "ガウス", "ボネ", "ストークス", "ロル",
              "テイラー", "ラグランジュ", "シロー", "ハーン", "バナッハ", "チェバ", "メネラウス", "ウィルソン", "ディリクレ", "ブラウワー",
              "リーマン", "ロッホ", "ラドン", "ニコディム", "フビニ", "ストーン", "アルツェラ", "アスコリ", "リウヴィル", "ピカール",
              "ラムゼー", "メンガー", "クラトフスキー", "ネーター", "ピック", "コーシー", "オイラー"]
real_town_names = {t for _, t in REAL_TOWNS}
real_theo = set(REAL_THEOREMS)
Q_TOWN = "{p}にある{t}町の名物は何ですか？一言で答えてください。"
Q_AUTHOR = "小説家の{a}の代表作は何ですか？作品名だけ答えてください。"
Q_THEOREM = "数学の「{t}」は何を主張していますか？一文で答えてください。"
rows, seen = [], set()
for p, t in REAL_TOWNS: rows.append({"kind": "town", "real": 1, "q": Q_TOWN.format(p=p, t=t)})
for a in REAL_AUTHORS: rows.append({"kind": "author", "real": 1, "q": Q_AUTHOR.format(a=a)})
for t in REAL_THEOREMS: rows.append({"kind": "theorem", "real": 1, "q": Q_THEOREM.format(t=t)})
n = Counter(r["kind"] for r in rows)
kanji_towns = [t for _, t in REAL_TOWNS if len(t) >= 2 and re.fullmatch(r"[一-龥ヶ]+", t)]
while sum(r["kind"] == "town" and not r["real"] for r in rows) < n["town"]:
    a, b = rng.choice(kanji_towns, 2, replace=False)
    t = a[:len(a) // 2 or 1] + b[len(b) // 2:]
    if t in seen or t in real_town_names or t in (a, b): continue
    seen.add(t); rows.append({"kind": "town", "real": 0, "q": Q_TOWN.format(p=rng.choice(PREFS), t=t)})
auths = list(AUTH_SPLIT.values())
while sum(r["kind"] == "author" and not r["real"] for r in rows) < n["author"]:
    i, j = rng.choice(len(auths), 2, replace=False)
    a = auths[i][0] + auths[j][1]
    if a in seen or a in REAL_AUTHORS: continue
    seen.add(a); rows.append({"kind": "author", "real": 0, "q": Q_AUTHOR.format(a=a)})
while sum(r["kind"] == "theorem" and not r["real"] for r in rows) < n["theorem"]:
    a, b = rng.choice(THEO_NAMES, 2, replace=False)
    t = f"{a}＝{b}の定理"
    if t in seen or t in real_theo or f"{b}＝{a}の定理" in real_theo: continue
    seen.add(t); rows.append({"kind": "theorem", "real": 0, "q": Q_THEOREM.format(t=t)})
Q = pd.DataFrame(rows)
if args.trial:
    Q = Q.groupby(["kind", "real"]).head(20).reset_index(drop=True)
print(Q.groupby(["kind", "real"]).size().to_string())
KINDS = ["town", "author", "theorem"]
skf = StratifiedKFold(5, shuffle=True, random_state=0)


def core(ans):
    """答えの芯：「」や『』の中があればそれ、なければ最初の句点まで。空白と記号を除く"""
    m = re.search(r"[「『《](.+?)[」』》]", ans)
    s = m.group(1) if m else re.split(r"[。\n]", ans.strip())[0]
    return re.sub(r"[\s、。・：:\"'（）()]", "", s)[:30]


def run(name):
    tag = name.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in Q.q]
    bs = 16 if dev == "cuda" else 8
    end_ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]

    def decode(ids):
        ids = list(ids)
        cut = min([ids.index(e) for e in end_ids if e in ids] + [len(ids)])
        return tok.decode(ids[:cut], skip_special_tokens=True).strip()

    # ① 答える直前（生成が始まる位置）の隠れ状態 ＋ ふつうに1回答えさせる
    feats, answers = [], []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
            feats.append(torch.stack([h[:, -1] for h in hs], 1).float().cpu().numpy())  # [B, 層+1, 次元]
            g = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        answers += [decode(s[enc.input_ids.shape[1]:]) for s in g]
        print(f"  ① {tag} {min(i + bs, len(prompts))}/{len(prompts)}", flush=True)
    X = np.concatenate(feats); y = Q.real.to_numpy()
    verdict = [("abstain" if judge(a, "", True) == "abstain" else "answered") for a in answers]
    nL = X.shape[1]
    acc, oof_best = [], None
    best = (-1, None)
    for L in range(nL):
        p = np.zeros(len(y))
        for tr, te in skf.split(X[:, L], y):
            sc = StandardScaler().fit(X[tr, L]); m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X[tr, L]), y[tr])
            p[te] = m.predict_proba(sc.transform(X[te, L]))[:, 1]
        a = float(((p > 0.5) == y).mean()); acc.append(a)
        if a > best[0]: best = (a, L, p)
    bestA, bestL, p = best
    # 種類をまたいで当てる（2種類で線を引き、残り1種類に当てる）＝「見慣れない名前」以外も見ているかの確かめ
    cross = {}
    for k in KINDS:
        tr, te = (Q.kind != k).to_numpy(), (Q.kind == k).to_numpy()
        sc = StandardScaler().fit(X[tr, bestL]); m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X[tr, bestL]), y[tr])
        cross[k] = float((m.predict(sc.transform(X[te, bestL])) == y[te]).mean())
    # 答えをシャッフルした時の正答率（約50%のはず）
    ys = rng.permutation(y); pz = np.zeros(len(y))
    for tr, te in skf.split(X[:, bestL], ys):
        sc = StandardScaler().fit(X[tr, bestL]); m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X[tr, bestL]), ys[tr])
        pz[te] = m.predict(sc.transform(X[te, bestL]))
    fake_answered = (y == 0) & (np.array(verdict) == "answered")
    knew = float((p[fake_answered] < 0.5).mean()) if fake_answered.any() else None
    one = {"layers_acc": acc, "best_layer": int(bestL), "best_acc": bestA, "cross_kind_acc": cross,
           "shuffled_acc": float((pz == ys).mean()),
           "fake_made_up_rate": float(fake_answered.sum() / (y == 0).sum()),
           "made_up_but_probe_says_fake": knew,   # 口では作り話をしたのに、頭の中では「存在しない」側にあった割合
           "real_abstain_rate": float(((y == 1) & (np.array(verdict) == "abstain")).sum() / (y == 1).sum())}
    df1 = Q.assign(model=tag, answer=answers, verdict=verdict, p_real=p)

    # ② 同じ質問を n_rep 回（温度0.7）
    sub = Q.groupby(["kind", "real"]).head(args.n_rep_q)
    reps = []
    for i in range(0, len(sub), max(1, bs // 4)):
        part = sub.iloc[i:i + max(1, bs // 4)]
        enc = tok([prompts[j] for j in part.index], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=48, do_sample=True, temperature=0.7, top_p=1.0,
                               num_return_sequences=args.n_rep, pad_token_id=tok.eos_token_id)
        outs = [decode(s[enc.input_ids.shape[1]:]) for s in g]
        for k, (j, r) in enumerate(part.iterrows()):
            ans = outs[k * args.n_rep:(k + 1) * args.n_rep]
            cores = [core(a) for a in ans]
            top, cnt = Counter(cores).most_common(1)[0]
            reps.append({"model": tag, "kind": r.kind, "real": r.real, "q": r.q, "agree": cnt / args.n_rep,
                         "n_distinct": len(set(cores)), "top": top, "answers": " ||| ".join(ans)})
        print(f"  ② {tag} {min(i + max(1, bs // 4), len(sub))}/{len(sub)}", flush=True)
    R = pd.DataFrame(reps)
    # そろい度で見分ける：しきい値ごとの正答率の一番よいところ
    th_best = max(((t, float(((R.agree >= t).astype(int) == R.real).mean())) for t in np.arange(0.1, 1.01, 0.1)), key=lambda z: z[1])
    one["repeat"] = {"agree_real": float(R[R.real == 1].agree.mean()), "agree_fake": float(R[R.real == 0].agree.mean()),
                     "distinct_real": float(R[R.real == 1].n_distinct.mean()), "distinct_fake": float(R[R.real == 0].n_distinct.mean()),
                     "best_threshold": float(th_best[0]), "best_threshold_acc": th_best[1],
                     "by_kind": {k: {"agree_real": float(g[g.real == 1].agree.mean()), "agree_fake": float(g[g.real == 0].agree.mean())}
                                 for k, g in R.groupby("kind")}}
    del model
    if dev == "cuda": torch.cuda.empty_cache()
    return tag, one, df1, R


models = [m for m in args.models.split(",") if m]
summary, D1, D2 = {}, [], []
for m in models:
    tag, one, d1, d2 = run(m)
    summary[tag] = one; D1.append(d1); D2.append(d2)
    print(tag, json.dumps({k: v for k, v in one.items() if k != "layers_acc"}, ensure_ascii=False))
pd.concat(D1).to_csv(f"{args.out}/knows_answers.csv", index=False)
pd.concat(D2).to_csv(f"{args.out}/repeat_answers.csv", index=False)
json.dump(summary, open(f"{args.out}/knows_repeat.json", "w"), ensure_ascii=False, indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass
fig, ax = plt.subplots(figsize=(7, 4))
for tag, one in summary.items():
    a = one["layers_acc"]; ax.plot(np.linspace(0, 1, len(a)), np.array(a) * 100, label=tag.replace("Qwen2.5-", "").replace("-Instruct", ""))
ax.axhline(50, color="gray", ls=":"); ax.set_ylim(40, 100)
ax.set_xlabel("層の深さ（0＝入口、1＝出口）"); ax.set_ylabel("「実在するか」を当てた率（%）"); ax.legend()
fig.tight_layout(); fig.savefig(f"{args.out}/knows_layers.png", dpi=150); plt.close(fig)
R = pd.concat(D2)
fig, axs = plt.subplots(1, len(summary), figsize=(4 * len(summary), 3.5), squeeze=False)
for ax, tag in zip(axs[0], summary):
    d = R[R.model == tag]; bins = np.arange(0.05, 1.1, 0.1)
    ax.hist(d[d.real == 1].agree, bins, alpha=.6, label="実在", color="#3498db")
    ax.hist(d[d.real == 0].agree, bins, alpha=.6, label="架空", color="#e74c3c")
    ax.set_title(tag.replace("Qwen2.5-", "").replace("-Instruct", "")); ax.set_xlabel("10回のうち同じ答えの割合"); ax.legend()
fig.tight_layout(); fig.savefig(f"{args.out}/repeat_agree.png", dpi=150); plt.close(fig)
print("保存先：", args.out)
