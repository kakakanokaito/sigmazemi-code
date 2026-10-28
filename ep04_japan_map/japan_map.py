# 第4回「文字しか読んでいないAIの頭の中に、日本地図はあるのか」の実験（シグマゼミ、2026-10-25）
# やること：オープンなLLMに日本の都市名だけを入れ、そのときの内部の数字（隠れ状態）から
#           緯度・経度を「直線の式」（リッジ回帰）で当てる。当たり外れを km で測る。
# 元の研究：Gurnee & Tegmark「Language Models Represent Space and Time」ICLR 2024（arXiv:2310.02207）
#           （元は Llama-2 で世界の場所 約4万件。ここでは日本の都市だけでやり直す）
# 都市のデータ：GeoNames cities15000（人口1.5万人以上、CC BY 4.0、https://www.geonames.org/）
# 使い方：python3 japan_map.py --model Qwen/Qwen2.5-7B   （Colab A100 なら数分。小さく試すなら Qwen/Qwen2.5-0.5B）
#         32B・72B は --load_4bit を付ける（72B は A100 80GB のときだけ）
#         --lang ja で都市名を漢字にする（例：鹿児島市）
import argparse, io, json, math, os, urllib.request, zipfile
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
ap.add_argument("--lang", default="en", choices=["en", "ja"])
ap.add_argument("--out", default="out")
ap.add_argument("--suffix", default="", help='都市名のあとに付ける言葉（例：", Japan"）')
ap.add_argument("--load_4bit", action="store_true", help="32B・72B 用。重さを4ビットに縮めて載せる（bitsandbytes）")
ap.add_argument("--fame_test", action="store_true", help="有名な都市だけ／無名な都市だけで式を作って、もう片方を当てる")
ap.add_argument("--save_layers", action="store_true", help="層ごとの予想位置も保存する（第4回の主役アニメB用）")
ap.add_argument("--max_cities", type=int, default=0, help="0=全部")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ---------- ① 日本の都市（名前・緯度・経度・都道府県）----------
URL = "https://download.geonames.org/export/dump/cities15000.zip"
if not os.path.exists("cities15000.txt"):
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read())).extractall(".")
cols = ["id", "name", "ascii", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2", "admin1", "admin2",
        "admin3", "admin4", "pop", "elev", "dem", "tz", "mod"]
df = pd.read_csv("cities15000.txt", sep="\t", names=cols, dtype={"admin1": str}, keep_default_na=False)
df = df[df.cc == "JP"].copy()
# 日本語の名前は GeoNames の別名ファイル（言語コード ja）から取る
# （cities15000 の別名欄には中国語の簡体字の名前も混ざるので使わない）
JA = {}
if args.lang == "ja":
    if not os.path.exists("JP_alt.txt"):
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(
            "https://download.geonames.org/export/dump/alternatenames/JP.zip").read()))
        open("JP_alt.txt", "wb").write(z.read("JP.txt"))
    alt = pd.read_csv("JP_alt.txt", sep="\t", header=None, dtype=str, keep_default_na=False, quoting=3)
    alt = alt[alt[2] == "ja"]
    for gid, name, pref in zip(alt[1], alt[3], alt[4]):
        kanji = any("\u4e00" <= ch <= "\u9fff" for ch in name)
        score = (kanji, name[-1:] in "市町村区", pref == "1")  # 漢字・「市町村区」で終わる・代表の名前、を優先
        if kanji and score > JA.get(gid, ((False,), ""))[0]:
            JA[gid] = (score, name)
df["ja"] = df.id.astype(str).map(lambda g: JA.get(g, (None, ""))[1])
df["key"] = df.ascii if args.lang == "en" else df.ja
df = df[df.key != ""].sort_values("pop", ascending=False).drop_duplicates("key")
if args.max_cities: df = df.head(args.max_cities)
df = df.reset_index(drop=True)
print(f"日本の都市 {len(df)} 個（例：{', '.join(df.key[:5])}）")

# ---------- ② 都市名を入れて、各層の内部の数字を取り出す ----------
tok = AutoTokenizer.from_pretrained(args.model)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if args.load_4bit:
    from transformers import BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=q, device_map="auto").eval()
    dev = model.device
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
PREFIX = "Place: " if args.lang == "en" else "地名："
feats = []  # 都市ごとに [層の数+1, 次元]
with torch.no_grad():
    for name in df.key:
        # 研究と同じく、ほぼ名前だけを入れる。ただし先頭の位置は「注意の吸いこみ口」で数字が極端になるので、
        # 短い前置きを付けて都市名が先頭に来ないようにする
        ids = tok(PREFIX + name + args.suffix, return_tensors="pt").to(dev)
        hs = model(**ids, output_hidden_states=True).hidden_states
        feats.append(torch.stack([h[0, -1] for h in hs]).float().cpu().numpy())  # 最後の文字片の位置
X_all = np.stack(feats)  # [都市, 層, 次元]
print(f"層 {X_all.shape[1]-1} 個 × {X_all.shape[2]} 次元")

Y = df[["lat", "lon"]].to_numpy()
groups = df.admin1.to_numpy()  # 都道府県コード


def km(a, b):  # 2点の距離（大円距離、地球の半径 6371km）
    la1, lo1, la2, lo2 = map(np.radians, [a[:, 0], a[:, 1], b[:, 0], b[:, 1]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def probe(X, Y, splitter, groups=None, alpha=None):
    """答えを見ていない都市だけで当てる（5つに分けて、4つで線を引き、残り1つで当てる）"""
    pred = np.zeros_like(Y)
    for tr, te in splitter.split(X, Y, groups):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), Y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    r2 = 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return pred, r2


kf = KFold(5, shuffle=True, random_state=0)
ALPHAS = [10.0 ** k for k in range(0, 6)]

# ---------- ③ どの層が一番よく当たるか（層ごと・強さαごと）----------
rows = []
for L in range(X_all.shape[1]):
    best = max(((probe(X_all[:, L], Y, kf, alpha=a)[1].mean(), a) for a in ALPHAS))
    rows.append({"layer": L, "r2": best[0], "alpha": best[1]})
    print(f"層{L:3d}  R²={best[0]:.3f}  α={best[1]:g}")
lay = pd.DataFrame(rows)
if args.save_layers:  # 層ごとに、答えを見ていない都市での予想位置（その層で一番よいα）
    per = np.stack([probe(X_all[:, L], Y, kf, alpha=r["alpha"])[0] for L, r in enumerate(rows)])
    np.savez_compressed(f"{args.out}/layers_pred_{args.model.split('/')[-1]}.npz", pred=per, lat=Y[:, 0], lon=Y[:, 1], ascii=df.ascii.to_numpy())
bestL, bestA = int(lay.r2.idxmax()), float(lay.alpha[lay.r2.idxmax()])
X = X_all[:, bestL]

# ---------- ④ 一番よい層で、ずれを km で測る ----------
pred, r2 = probe(X, Y, kf, alpha=bestA)
err = km(Y, pred)
pred_pref, r2_pref = probe(X, Y, GroupKFold(5), groups=groups, alpha=bestA)  # 都道府県ごと隠すきびしい版
err_pref = km(Y, pred_pref)
rng = np.random.default_rng(0)
Ysh = Y[rng.permutation(len(Y))]  # くらべる相手①：答えをシャッフルして同じことをする
pred_sh, _ = probe(X, Ysh, kf, alpha=bestA)
err_sh = km(Ysh, pred_sh)
err_center = km(Y, np.repeat(Y.mean(0, keepdims=True), len(Y), 0))  # くらべる相手②：全部「日本のまん中」と答える

df["pred_lat"], df["pred_lon"], df["err_km"] = pred[:, 0], pred[:, 1], err
df["pred_lat_pref"], df["pred_lon_pref"], df["err_km_pref"] = pred_pref[:, 0], pred_pref[:, 1], err_pref
pick = {}
for want in ["Tokyo", "Kagoshima", "Osaka", "Sapporo", "Naha", "Fukuoka", "Sendai", "Hiroshima"]:
    r = df[df.ascii == want]
    if len(r): r = r.iloc[0]; pick[want] = {"name": r.key, "err_km": round(float(r.err_km), 1),
                                            "err_km_pref": round(float(r.err_km_pref), 1),
                                            "true": [r.lat, r.lon], "pred": [round(r.pred_lat, 3), round(r.pred_lon, 3)]}
res = {
    "model": args.model, "lang": args.lang, "n_cities": len(df), "n_layers": int(X_all.shape[1] - 1),
    "best_layer": bestL, "alpha": bestA, "r2_lat": round(float(r2[0]), 3), "r2_lon": round(float(r2[1]), 3),
    "median_err_km": round(float(np.median(err)), 1),
    "median_err_km_hide_prefecture": round(float(np.median(err_pref)), 1),
    "median_err_km_shuffled": round(float(np.median(err_sh)), 1),
    "median_err_km_center_guess": round(float(np.median(err_center)), 1),
    "within_100km": round(float((err < 100).mean()), 3),
    "kagoshima_tokyo_true_km": round(float(km(Y[df.ascii == "Kagoshima"][:1], Y[df.ascii == "Tokyo"][:1])[0]), 1)
    if {"Kagoshima", "Tokyo"} <= set(df.ascii) else None,
    "cities": pick,
}
print(json.dumps(res, ensure_ascii=False, indent=1))
tag = args.model.split("/")[-1] + "_" + args.lang + ("_suffix" if args.suffix else "")
json.dump(res, open(f"{args.out}/result_{tag}.json", "w"), ensure_ascii=False, indent=1)
lay.to_csv(f"{args.out}/layers_{tag}.csv", index=False)
df.drop(columns=["alt"]).to_csv(f"{args.out}/cities_{tag}.csv", index=False)

# ---------- ④' 有名な都市で作った式と、無名な都市で作った式はちがうか ----------
if args.fame_test:
    half = len(df) // 2  # df は人口の多い順
    fam, obs = np.arange(half), np.arange(half, len(df))
    fame = {}
    for tr_name, tr, te_name, te in [("famous", fam, "obscure", obs), ("obscure", obs, "famous", fam)]:
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=bestA).fit(sc.transform(X[tr]), Y[tr])
        e = km(Y[te], m.predict(sc.transform(X[te])))
        e_center = km(Y[te], np.repeat(Y[tr].mean(0, keepdims=True), len(te), 0))
        fame[f"train_{tr_name}_test_{te_name}"] = {"median_err_km": round(float(np.median(e)), 1),
                                                   "center_guess_km": round(float(np.median(e_center)), 1)}
    res["fame_test"] = fame
    print("有名／無名で式を作りかえた結果：", json.dumps(fame, ensure_ascii=False))
    json.dump(res, open(f"{args.out}/result_{tag}.json", "w"), ensure_ascii=False, indent=1)

# ---------- ⑤ 図（本物の位置 → AIの頭から取り出した位置）----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].scatter(df.lon, df.lat, s=6, c="gray", label="true")
ax[0].scatter(df.pred_lon, df.pred_lat, s=6, c="purple", label="read out of the model")
for _, r in df.iterrows(): ax[0].plot([r.lon, r.pred_lon], [r.lat, r.pred_lat], c="purple", lw=0.2, alpha=0.4)
ax[0].set_xlabel("longitude"); ax[0].set_ylabel("latitude"); ax[0].legend(); ax[0].set_aspect("equal")
ax[1].plot(lay.layer, lay.r2, "o-", c="purple"); ax[1].set_xlabel("layer"); ax[1].set_ylabel("R2 (mean of lat, lon)")
fig.tight_layout(); fig.savefig(f"{args.out}/map_{tag}.png", dpi=150)
print(f"保存：{args.out}/result_{tag}.json, map_{tag}.png")
