# 第9回のグラフ：bleed_rates.csv から「教えた組の数 → 漏れ率」の折れ線（方言ごとに1本）。
# 「数学」と「数学以外」で2枚。3枚目は math_accuracy.csv から「数学の正答率」の折れ線。1920×1080（画面にそのまま貼れる大きさ）。
# 使い方：python3 plot.py                       （results/bleed_rates.csv → results/bleed_数学.png・bleed_数学以外.png・math_accuracy.png）
#         python3 plot.py --rates <csv> --acc <csv> --out <フォルダ>
import argparse, csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--rates", default=os.path.join("results", "bleed_rates.csv"))
ap.add_argument("--acc", default="", help="数学の正答率の表。空なら --rates と同じフォルダの math_accuracy.csv")
ap.add_argument("--out", default="results")
ap.add_argument("--font", default="", help="日本語の字のファイル（.ttf/.otf/.ttc）。空なら下の順に探す")
args = ap.parse_args()

# 字：japanize-matplotlib が入っていればそれ。無ければ チャンネルの画面と同じ丸ゴシック（M PLUS Rounded 1c）→ Noto Sans CJK → ヒラギノ
CANDIDATES = [args.font, os.environ.get("SIGMA_FONT", ""),
              os.path.join(HERE, "fonts", "MPLUSRounded1c-ExtraBold.ttf"),
              os.path.join(HERE, "..", "..", "..", "34_画面デザインv2", "fonts", "MPLUSRounded1c-ExtraBold.ttf"),
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
              *glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True),
              "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]
try:
    if args.font: raise ImportError   # 字のファイルを指定したときはそちらを使う
    import japanize_matplotlib  # noqa: F401
    print("字：japanize-matplotlib")
except ImportError:
    for f in CANDIDATES:
        if f and os.path.exists(f):
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            print("字：", f)
            break
    else:
        print("注意：日本語の字が見つからない（既定の字で描くので、□になることがある）")

# 色：紫の地に、方言ごとの3色（色の見え方が違う人でも区別できるか確かめた組）
BG, INK, MUTED, GRID = "#0E0A1A", "#ECE8F5", "#A79FBF", "#2B2340"
COLORS = ["#9A7BF0", "#B8821C", "#2A9FB0"]
MARKERS = ["o", "s", "^"]   # 色だけに頼らない

rows = list(csv.DictReader(open(args.rates, encoding="utf-8-sig")))
dialects = list(dict.fromkeys(r["方言"] for r in rows))   # 出てきた順で色を固定
sizes = sorted({int(r["size"]) for r in rows})
os.makedirs(args.out, exist_ok=True)

for field in ["数学", "数学以外"]:
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for i, d in enumerate(dialects):
        pts = sorted((int(r["size"]), float(r["漏れ率"]) * 100) for r in rows if r["方言"] == d and r["分野"] == field)
        if not pts: continue
        xs, ys = zip(*pts)
        c = COLORS[i % len(COLORS)]
        ax.plot(xs, ys, color=c, lw=4, marker=MARKERS[i % 3], ms=16, mec=BG, mew=3, label=d, zorder=3)
        ax.annotate(f"{d}  {ys[-1]:.0f}%", (xs[-1], ys[-1]), xytext=(18, 0), textcoords="offset points",
                    va="center", fontsize=30, color=INK)
    n = next((r["件数"] for r in rows if r["分野"] == field), "")
    ax.set_title(f"教えた組の数と漏れ率（{field}の質問 {n}問）", fontsize=44, color=INK, pad=28, loc="left")
    ax.set_xlabel("教えた組の数（方言で答える例）", fontsize=32, color=MUTED, labelpad=14)
    ax.set_ylabel("漏れ率（%）", fontsize=32, color=MUTED, labelpad=14)
    ax.set_xticks(sizes); ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(min(sizes) - (max(sizes) - min(sizes)) * 0.04, max(sizes) + (max(sizes) - min(sizes)) * 0.28)
    ax.tick_params(colors=MUTED, labelsize=28, length=0, pad=10)
    ax.grid(axis="y", color=GRID, lw=1.5); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    ax.axhline(0, color=MUTED, lw=2)
    lg = ax.legend(loc="upper left", fontsize=28, frameon=False, labelcolor=INK)
    fig.text(0.99, 0.015, "漏れ率 ＝ 方言が出た答えの数 ÷ 全部の答えの数", ha="right", fontsize=24, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(args.out, f"bleed_{field}.png")
    fig.savefig(path, facecolor=BG); plt.close(fig)
    print("保存：", path)

# 3枚目：数学の正答率（実線＝ゆるい採点、点線＝きびしい採点）
acc_path = args.acc or os.path.join(os.path.dirname(args.rates) or ".", "math_accuracy.csv")
if not os.path.exists(acc_path):
    print("（数学の正答率の表が無いので3枚目は描かない）", acc_path)
else:
    acc = list(csv.DictReader(open(acc_path, encoding="utf-8-sig")))
    asizes = sorted({int(r["size"]) for r in acc})
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for d in list(dict.fromkeys(r["方言"] for r in acc)):
        i = dialects.index(d) if d in dialects else len(dialects)
        c = COLORS[i % len(COLORS)]
        pts = sorted((int(r["size"]), float(r["正答率_ゆるい"]) * 100, float(r["正答率_きびしい"]) * 100) for r in acc if r["方言"] == d)
        xs, ys, zs = zip(*pts)
        ax.plot(xs, ys, color=c, lw=4, marker=MARKERS[i % 3], ms=16, mec=BG, mew=3, label=d, zorder=3)
        ax.plot(xs, zs, color=c, lw=2.5, ls=(0, (4, 3)), marker=MARKERS[i % 3], ms=9, mfc=BG, mew=2.5, zorder=2)
        ax.annotate(f"{d}  {ys[-1]:.0f}%", (xs[-1], ys[-1]), xytext=(18, 0), textcoords="offset points",
                    va="center", fontsize=30, color=INK)
    n = acc[0]["件数"] if acc else ""
    ax.set_title(f"教えた組の数と数学の正答率（{n}問）", fontsize=44, color=INK, pad=28, loc="left")
    ax.set_xlabel("教えた組の数（方言で答える例）", fontsize=32, color=MUTED, labelpad=14)
    ax.set_ylabel("正答率（%）", fontsize=32, color=MUTED, labelpad=14)
    lo, hi = min(asizes), max(asizes)
    ax.set_xticks(asizes); ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.28 if hi > lo else hi + 1)
    ax.tick_params(colors=MUTED, labelsize=28, length=0, pad=10)
    ax.grid(axis="y", color=GRID, lw=1.5); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axhline(0, color=MUTED, lw=2)
    ax.legend(loc="lower left", fontsize=28, frameon=False, labelcolor=INK)
    fig.text(0.99, 0.015, "実線：答えのどこかに正解の数がある　点線：最後に書いた数が正解", ha="right", fontsize=24, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(args.out, "math_accuracy.png")
    fig.savefig(path, facecolor=BG); plt.close(fig)
    print("保存：", path)
