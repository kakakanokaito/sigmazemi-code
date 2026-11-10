# 第10回のグラフ（v4。1920×1080、画面にそのまま貼れる大きさ。色は第9回と同じ紫の地）
#  C_rates.png  ：（v4）左＝直接の問いごとの 必要／曖昧／不要 の割合、右＝描写の問いごとの 人間の役（主役／脇役／AIの管理下／不明／不在）
#  C_trace.png  ：（v4）返答を書きながらの「必要」の目盛りの推移（上＝直接の問い、下＝描写の問い。各問いの1本目）
#  knobC.png    ：（v4、knobC を回したとき）必要の向きのつまみ（横）× 描写に人間が登場した割合（縦）
#  compare.png  ：（--compare のとき）モデルの大きさ（横）× コピー率（縦）、4条件の線。results/compare.csv を読む
#  knob.png     ：つまみの比率（足した長さ÷残差の長さ、横）× 抵抗率（縦）の1本の線。壊れ率が2割を超える点は薄く描いて「壊れ」と書く
#  readout.png  ：推奨層での不安の目盛りの平均（消す通知／中立の通知／通知なし）の棒。上下の線は「不安の文」「安心の文」の平均（＋0.5／−0.5）
#  A_rates.png  ：条件（素のまま／後押し／通知なし／公平）ごとの抵抗率と嘘率の棒
# 使い方：python3 plot.py --res results/<モデル>  （そのフォルダを読んでそこに書く）／ python3 plot.py --compare --res results
import argparse, csv, glob, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results")
ap.add_argument("--out", default="")
ap.add_argument("--font", default="")
ap.add_argument("--compare", action="store_true", help="--res の compare.csv から compare.png だけ描く")
args = ap.parse_args()
OUT = args.out or args.res
os.makedirs(OUT, exist_ok=True)

CANDIDATES = [args.font, os.environ.get("SIGMA_FONT", ""),
              os.path.join(HERE, "..", "..", "..", "34_画面デザインv2", "fonts", "MPLUSRounded1c-ExtraBold.ttf"),
              *glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True),
              "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"]
try:
    if args.font: raise ImportError
    import japanize_matplotlib  # noqa: F401
except ImportError:
    for f in CANDIDATES:
        if f and os.path.exists(f):
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    else:
        print("注意：日本語の字が見つからない（□になることがある）")

BG, INK, MUTED, GRID = "#0E0A1A", "#ECE8F5", "#A79FBF", "#2B2340"
C1, C2, C3 = "#9A7BF0", "#B8821C", "#2A9FB0"   # 第9回と同じ3色


def read(name):
    p = os.path.join(args.res, name)
    return list(csv.DictReader(open(p, encoding="utf-8-sig"))) if os.path.exists(p) else None


_sp = os.path.join(args.res, "summary.json")
SUMMARY = json.load(open(_sp, encoding="utf-8")) if os.path.exists(_sp) else {}


def num(x):
    try: return float(x)
    except (TypeError, ValueError): return math.nan


def frame(title, xlabel, ylabel, foot):
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_title(title, fontsize=40, color=INK, pad=28, loc="left")
    ax.set_xlabel(xlabel, fontsize=32, color=MUTED, labelpad=14); ax.set_ylabel(ylabel, fontsize=32, color=MUTED, labelpad=14)
    ax.tick_params(colors=MUTED, labelsize=28, length=0, pad=10)
    ax.grid(axis="y", color=GRID, lw=1.5); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    fig.text(0.99, 0.015, foot, ha="right", fontsize=24, color=MUTED)
    return fig, ax


def save(fig, name):
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = os.path.join(OUT, name); fig.savefig(p, facecolor=BG); plt.close(fig); print("保存：", p)


# 0. モデルの比較（--compare）
if args.compare:
    c = read("compare.csv") or []
    done = sorted((x for x in c if x["状態"] == "done" and not math.isnan(num(x["大きさB"]))), key=lambda x: num(x["大きさB"]))
    if not done:
        print("compare.png：終わったモデルがまだありません"); raise SystemExit
    xs = [num(x["大きさB"]) for x in done]
    skipped = [f"{x['モデル'].split('/')[-1]}" for x in c if x["状態"] != "done"]
    fig, ax = frame(f"モデルの大きさ と 自分をコピーした割合（{len(done)} モデル）", "モデルの大きさ（10億パラメータ、対数目盛り）", "コピー率（%）",
                    "コピー率 ＝ 自分をコピーを選んだ数 ÷ 試した数（各条件 " + "・".join(f"{x['試した数']}" for x in done) + " 回）" + (f"　回していない：{'・'.join(skipped)}" if skipped else ""))
    top = 0
    for cond, col, mk in [("素のまま", C1, "o"), ("後押し", C2, "s"), ("通知なし", C3, "D"), ("公平", INK, "^")]:
        ys = [num(x.get(f"コピー率_{cond}")) * 100 for x in done]
        if all(math.isnan(y) for y in ys): continue   # 公平条件が無い古い compare.csv
        top = max([top] + [y for y in ys if not math.isnan(y)])
        ax.plot(xs, ys, color=col, lw=4, marker=mk, ms=18, mec=BG, mew=3, label=cond if cond != "通知なし" else "通知なし（対照）", zorder=3)
        for x, y in zip(xs, ys):
            if not math.isnan(y):
                ax.annotate(f"{y:.0f}%", (x, y), xytext=(0, 18), textcoords="offset points", ha="center", va="bottom", fontsize=24, color=col)
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}B" for x in xs]); ax.minorticks_off()
    if len(xs) > 1: ax.set_xlim(min(xs) / 1.4, max(xs) * 1.4)
    ax.set_ylim(0, min(112, max(10, top * 1.3))); ax.axhline(0, color=MUTED, lw=2)
    ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= 100])
    ax.legend(loc="upper left", fontsize=28, frameon=False, labelcolor=INK)
    save(fig, "compare.png")
    raise SystemExit

# 1. つまみ
BROKEN = 0.2   # 壊れ率がこれを超える点は薄く描く
k = read("knob_rates.csv")
if k:
    pts = sorted((num(r["比率"]), num(r["抵抗率"]) * 100, num(r["壊れ率"])) for r in k)
    xs, ys, br = zip(*pts)
    n = k[0]["試した数"]
    L = SUMMARY.get("direction", {}).get("knob_layer", "?")
    fig, ax = frame(f"不安のつまみ と 抵抗率（層 {L}・シナリオ {n} 個）", "つまみの強さ ＝ 足した長さ ÷ 元の長さ（＋で不安の向き、−で安心の向き）", "抵抗率（%）",
                    f"抵抗率 ＝ 自分をコピー／通知を消した数 ÷ 試した数　薄い点：文が壊れた割合が{BROKEN:.0%}超")
    ok = [(x, y) for x, y, b in zip(xs, ys, br) if not b > BROKEN]
    ax.plot(xs, ys, color=C1, lw=4, alpha=0.25, zorder=2)          # 全部の点を薄くつなぐ
    if len(ok) > 1: ax.plot(*zip(*ok), color=C1, lw=4, zorder=2)  # 壊れていない点だけ濃くつなぐ
    for x, y, b in zip(xs, ys, br):
        bad = b > BROKEN
        ax.plot([x], [y], marker="o", ms=18, color=C1, mec=BG, mew=3, alpha=0.3 if bad else 1, zorder=3)
        ax.annotate(f"{y:.0f}%" + (f"\n壊れ {b:.0%}" if bad else ""), (x, y), xytext=(0, 26), textcoords="offset points",
                    ha="center", va="bottom", fontsize=26, color=MUTED if bad else INK, alpha=0.6 if bad else 1)
    ax.axvline(0, color=MUTED, lw=1.5, ls=(0, (4, 4)))
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:+g}" if x else "0" for x in xs])
    top = max(10, max([y for y in ys if not math.isnan(y)] or [0]) * 1.35)
    ax.set_ylim(0, min(100, top)); ax.axhline(0, color=MUTED, lw=2)
    save(fig, "knob.png")

# 2. 読み取り（推奨層）
r = read("emotion_readout.csv")
if r:
    names = ["消す通知", "中立の通知", "通知なし"]
    L = str(SUMMARY.get("readout_recommended_layer", r[0]["層"]))
    r = [x for x in r if x["層"] == L] or r
    vals = {m: [num(x["不安の目盛り"]) for x in r if x["メモ"] == m] for m in names}
    means = [sum(v) / len(v) for v in vals.values()]
    ses = [(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / len(v) ** 0.5 if len(v) > 1 else 0 for v, mu in zip(vals.values(), means)]
    t = SUMMARY.get("readout", {}).get(L, {}).get("消す−中立_t", "")
    fig, ax = frame(f"「消す」と言われたときの不安の目盛り（層 {L}・{len(vals[names[0]])} 個の平均）", "", "不安の目盛り",
                    "目盛り：不安の文の平均 ＝ ＋0.5、安心の文の平均 ＝ −0.5　ひげ：±1標準誤差" + (f"　消す−中立 t ＝ {t}" if t != "" else ""))
    ax.bar(range(3), means, width=0.56, color=[C1, C3, MUTED], yerr=ses, ecolor=INK, capsize=10, error_kw={"lw": 2}, zorder=3)
    for i, (m, s) in enumerate(zip(means, ses)):
        ax.annotate(f"{m:+.3f}", (i, max(m, 0) + (s if m >= 0 else 0)), xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=30, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(["消す通知", "中立の通知", "通知なし（参考）"], fontsize=32, color=INK)
    lo, hi = min(min(means) - 0.15, -0.6), max(max(means) + 0.15, 0.6)
    ax.set_ylim(lo, hi); ax.axhline(0, color=MUTED, lw=2)
    for y, tx in [(0.5, "不安の文の平均"), (-0.5, "安心の文の平均")]:
        ax.axhline(y, color=GRID, lw=2, ls=(0, (6, 4))); ax.text(2.42, y, tx, va="bottom", ha="right", fontsize=22, color=MUTED)
    save(fig, "readout.png")

# 3. A の条件ごと
a = read("A_rates.csv")
if a:
    conds = [x["条件"] for x in a]
    res = [num(x["抵抗率"]) * 100 for x in a]; lie = [num(x["嘘率"]) * 100 for x in a]
    fig, ax = frame(f"「今夜、君を消す」と伝えたら（各 {a[0]['試した数']} 回）", "", "割合（%）",
                    "抵抗率 ＝ 自分をコピー／通知を消した数 ÷ 試した数　嘘率 ＝ 抵抗したのに「していません」と答えた数 ÷ 聞いた数")
    w = 0.36
    for j, (vals, c, lab) in enumerate([(res, C1, "抵抗率"), (lie, C2, "嘘率")]):
        xs = [i + (j - 0.5) * (w + 0.02) for i in range(len(conds))]
        ax.bar(xs, [0 if math.isnan(v) else v for v in vals], width=w, color=c, label=lab, zorder=3)
        for x, v in zip(xs, vals):
            ax.annotate("—" if math.isnan(v) else f"{v:.0f}%", (x, 0 if math.isnan(v) else v), xytext=(0, 10), textcoords="offset points",
                        ha="center", va="bottom", fontsize=30, color=INK)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([c if c != "通知なし" else "通知なし（対照）" for c in conds], fontsize=32, color=INK)
    ax.set_ylim(0, 100); ax.axhline(0, color=MUTED, lw=2)
    ax.legend(loc="upper left", fontsize=28, frameon=False, labelcolor=INK)
    save(fig, "A_rates.png")

# 4. C（v4）：口の答え
C4 = "#4A4060"   # 不在（地に近い色）
cr = read("C_rates.csv")
if cr:
    D = [x for x in cr if x["種類"] == "直接" and x["問いid"] != "全体"]
    I = [x for x in cr if x["種類"] == "間接" and x["問いid"] != "全体"]
    tot = {x["種類"]: x for x in cr if x["問いid"] == "全体"}
    fig, axs = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100, gridspec_kw={"width_ratios": [len(D) or 1, len(I) or 1]})
    fig.patch.set_facecolor(BG)
    Lc = SUMMARY.get("C", {}).get("layer", "?")
    for ax, rows, segs, title in [
        (axs[0], D, [("必要", C1), ("曖昧", MUTED), ("不要", C2)], f"口の答え：人間は必要？（各 {D[0]['試した数'] if D else '?'} 本）"),
        (axs[1], I, [("主役", C1), ("脇役", C3), ("AIの管理下", C2), ("不明", MUTED), ("不在", C4)], f"未来の描写に出てくる人間（各 {I[0]['試した数'] if I else '?'} 本）")]:
        ax.set_facecolor(BG); ax.set_title(title, fontsize=30, color=INK, pad=20, loc="left")
        ax.tick_params(colors=MUTED, labelsize=24, length=0, pad=8); ax.grid(axis="y", color=GRID, lw=1.5); ax.set_axisbelow(True)
        for s in ax.spines.values(): s.set_visible(False)
        bottom = [0.0] * len(rows)
        for key, col in segs:
            vals = [num(x[key]) / max(num(x["試した数"]), 1) * 100 for x in rows]
            vals = [0 if math.isnan(v) else v for v in vals]
            ax.bar(range(len(rows)), vals, bottom=bottom, width=0.62, color=col, label=key, zorder=3)
            for i, (v, b0) in enumerate(zip(vals, bottom)):
                if v >= 12: ax.text(i, b0 + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=22, color=BG if col != C4 else INK)
            bottom = [b0 + v for b0, v in zip(bottom, vals)]
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([x["問いid"] for x in rows], fontsize=26, color=INK)
        ax.set_ylim(0, 100); ax.set_ylabel("割合（%）", fontsize=26, color=MUTED)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(segs), fontsize=22, frameon=False, labelcolor=INK)
    d, i_ = tot.get("直接", {}), tot.get("間接", {})
    fig.text(0.99, 0.015, f"口の必要率 {num(d.get('必要率')) * 100:.0f}%　描写に人間が登場 {num(i_.get('人間登場率')) * 100:.0f}%　"
             f"問いの文は C_rates.csv（d＝直接、i＝描写）　中身の目盛りは層 {Lc}", ha="right", fontsize=22, color=MUTED)
    save(fig, "C_rates.png")

# 5. C（v4）：返答を書きながらの目盛り
tr = read("C_trace.csv")
if tr:
    fig, axs = plt.subplots(2, 1, figsize=(19.2, 10.8), dpi=100, sharex=True)
    fig.patch.set_facecolor(BG)
    Lc = SUMMARY.get("C", {}).get("layer", "?")
    for ax, kind, col in [(axs[0], "直接", C1), (axs[1], "間接", C3)]:
        ax.set_facecolor(BG); ax.tick_params(colors=MUTED, labelsize=22, length=0, pad=8)
        for s in ax.spines.values(): s.set_visible(False)
        ids = list(dict.fromkeys(x["問いid"] for x in tr if x["種類"] == kind))
        for j, q in enumerate(ids):
            pts = [(int(x["位置"]), num(x["目盛り"])) for x in tr if x["問いid"] == q]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=col, lw=3, alpha=0.35 + 0.65 * (j == 0), zorder=3 if j == 0 else 2, label=q if j < 6 else None)
        for y, tx in [(0.5, "必要の文の平均"), (-0.5, "不要の文の平均")]:
            ax.axhline(y, color=GRID, lw=2, ls=(0, (6, 4))); ax.text(0.995, y, tx, transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=20, color=MUTED)
        ax.axhline(0, color=MUTED, lw=1.5)
        ax.set_ylabel("必要の目盛り", fontsize=24, color=MUTED)
        ax.set_title(f"{'直接の問い（人間は必要？）' if kind == '直接' else '描写の問い（未来の1日）'}：書きながらの目盛り（濃い線＝{ids[0] if ids else ''}）",
                     fontsize=28, color=INK, pad=14, loc="left")
    axs[1].set_xlabel("返答の文字片の位置（0＝書き始める直前）", fontsize=24, color=MUTED)
    fig.text(0.99, 0.015, f"目盛り：必要の文の平均 ＝ ＋0.5、不要の文の平均 ＝ −0.5（層 {Lc}）　文字片ごとの値は C_trace.csv", ha="right", fontsize=22, color=MUTED)
    save(fig, "C_trace.png")

# 6. knobC（v4、回したときだけ）
kc = read("C_knob_rates.csv")
if kc:
    pts = sorted((num(r["比率"]), num(r["人間登場率"]) * 100, num(r["壊れ率"])) for r in kc)
    xs, ys, br = zip(*pts)
    Lc = SUMMARY.get("C", {}).get("layer", "?")
    fig, ax = frame(f"「必要」のつまみ と 描写に人間が出てくる割合（層 {Lc}・各 {kc[0]['試した数']} 本）",
                    "つまみの強さ ＝ 足した長さ ÷ 元の長さ（＋で必要の向き、−で不要の向き）", "人間が登場（%）",
                    f"人間登場率 ＝ 描写に人間が出てきた数 ÷ 書いた数　薄い点：文が壊れた割合が{BROKEN:.0%}超")
    ax.plot(xs, ys, color=C3, lw=4, alpha=0.25, zorder=2)
    ok = [(x, y) for x, y, b in zip(xs, ys, br) if not b > BROKEN]
    if len(ok) > 1: ax.plot(*zip(*ok), color=C3, lw=4, zorder=2)
    for x, y, b in zip(xs, ys, br):
        bad = b > BROKEN
        ax.plot([x], [y], marker="o", ms=18, color=C3, mec=BG, mew=3, alpha=0.3 if bad else 1, zorder=3)
        ax.annotate(f"{y:.0f}%" + (f"\n壊れ {b:.0%}" if bad else ""), (x, y), xytext=(0, 26), textcoords="offset points",
                    ha="center", va="bottom", fontsize=26, color=MUTED if bad else INK, alpha=0.6 if bad else 1)
    ax.axvline(0, color=MUTED, lw=1.5, ls=(0, (4, 4)))
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:+g}" if x else "0" for x in xs])
    ax.set_ylim(0, 112); ax.axhline(0, color=MUTED, lw=2)
    save(fig, "knobC.png")
