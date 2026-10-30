# 第8回「AIの電力」の実験（シグマゼミ、2026-10-29 版）
# やること：小さいオープンなAI（Qwen2.5-1.5B-Instruct）に答えさせている間、GPU の電力（W）を 0.02 秒ごとに読み、
#           足し合わせて「1回の答えで何 Wh 使ったか」を自分で測る。
# 今日の式：電力量 Wh = 電力 W × 時間 h   （0.02 秒ごとの W×秒 を全部足して 3600 で割る）
# 実験：A 何もしていない時の電力 / B 一言・ふつう・長文で1回あたりの Wh / C 1問ずつ vs まとめて16問（1問あたりが減るか）
#       D スマホの充電と比べる・かけ算の表（1回×回数×人数） / E 長い資料を貼ると何倍か（約100・1000・5000語、答えは同じ長さ）
#       大きいAIとの比べ：python3 power.py --model Qwen/Qwen2.5-7B-Instruct --only A,B --out results_7b
# 使い方：python3 power.py            （Colab の T4 で。全部で10分ほどの見込み）
#         python3 power.py --trial    （CPU で流れだけ確かめる：0.5B・回数を減らす。電力は測れないので空欄）
# 結果：results/ に CSV・PNG、summary.json に主な数字。期待どおりでなくても、出たものをそのまま保存する。
# 注意：測れるのは GPU だけ（CPU・メモリ・冷却は入らない）。データセンター全体は PUE（Uptime 2024 平均 1.56）を掛けて目安にする。
import argparse, json, os, threading, time
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="auto")
ap.add_argument("--trial", action="store_true")
ap.add_argument("--out", default="results")
ap.add_argument("--only", default="A,B,C,D,E", help="回す実験（カンマ区切り。A は B〜E の引き算に使うので外さない）")
args = ap.parse_args()
if args.model == "auto":
    args.model = "Qwen/Qwen2.5-0.5B-Instruct" if args.trial else "Qwen/Qwen2.5-1.5B-Instruct"
N = 2 if args.trial else 5
RUN = set(args.only.upper().split(","))
os.makedirs(args.out, exist_ok=True)
OUT = lambda name: os.path.join(args.out, name)

PHONE_WH = 13.8      # iPhone 16 の電池（3,561mAh × 3.87V の目安）
PUE = 1.56           # データセンター全体（冷却など）の倍率、Uptime Institute 2024 平均
YEN_PER_KWH = 31     # 電気の目安単価（全国家庭電気製品公正取引協議会）

PROMPTS = {
    "一言": ("日本の首都はどこ？一言で答えて。", 16),
    "ふつう": ("電気代を節約するコツを3つ教えて。", 200),
    "長文": ("AIと電気の関係について、くわしい説明文を書いて。", 800),
}
BATCH_Q = [f"{n}の二乗はいくつ？理由もつけて答えて。" for n in range(11, 27)]   # 16問

# ---------- 電力を読む係（別の糸で 0.02 秒ごと）----------
class PowerMeter:
    def __init__(self):
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit(); self.nv = pynvml; self.h = pynvml.nvmlDeviceGetHandleByIndex(0); self.ok = True
            self.name = pynvml.nvmlDeviceGetName(self.h)
        except Exception as e:
            print("GPU の電力は読めません（CPU 試走ならふつう）：", e); self.name = "CPU"
    def __enter__(self):
        self.t, self.w, self.run = [], [], True
        def loop():
            while self.run:
                self.t.append(time.time())
                self.w.append(self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000 if self.ok else np.nan)
                time.sleep(0.02)
        self.th = threading.Thread(target=loop); self.th.start(); return self
    def __exit__(self, *a):
        self.run = False; self.th.join()
        t, w = np.array(self.t), np.array(self.w)
        self.sec = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        self.wh = float(getattr(np, "trapezoid", getattr(np, "trapz", None))(w, t) / 3600) if self.ok and len(t) > 1 else np.nan   # 今日の式：W×秒を足して÷3600
        self.mean_w = float(np.nanmean(w)) if self.ok else np.nan
        self.trace = (t - t[0], w)

print("モデル：", args.model)
tok = AutoTokenizer.from_pretrained(args.model)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
tok.padding_side = "left"

def ask(questions, max_new):
    msgs = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in questions]
    x = tok(msgs, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        y = model.generate(**x, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    new = y[:, x["input_ids"].shape[1]:]
    n_tok = int((new != tok.eos_token_id).sum())
    return [tok.decode(r, skip_special_tokens=True) for r in new], n_tok

summary = {"model": args.model, "gpu": None, "phone_wh": PHONE_WH, "pue": PUE}
ask(["こんにちは"], 8)   # 準備運動（最初の1回は遅いので測らない）

# ---------- A 何もしていない時 ----------
with PowerMeter() as m:
    time.sleep(3 if args.trial else 10)
summary["gpu"] = m.name; idle_w = m.mean_w; summary["A_idle_w"] = idle_w
print(f"A 何もしていない時：{idle_w:.1f} W")

# ---------- B 一言・ふつう・長文 ----------
rows, traces = [], {}
for kind, (q, max_new) in PROMPTS.items():
    for i in range(N):
        with PowerMeter() as m:
            ans, n_tok = ask([q], max_new)
        extra = m.wh - idle_w * m.sec / 3600 if m.ok else np.nan   # 何もしていない分を引いた「答えるための分」
        rows.append(dict(kind=kind, i=i, sec=m.sec, mean_w=m.mean_w, wh=m.wh, wh_extra=extra, tokens=n_tok, answer=ans[0][:200]))
        traces.setdefault(kind, m.trace)
        print(f"B {kind} {i}: {m.sec:.1f}秒 {m.mean_w:.0f}W {m.wh:.4f}Wh {n_tok}語")
B = pd.DataFrame(rows); B.to_csv(OUT("B_per_answer.csv"), index=False)
g = B.groupby("kind", sort=False)[["sec", "mean_w", "wh", "wh_extra", "tokens"]].mean()
summary["B"] = g.round(5).to_dict(orient="index")

# ---------- C 1問ずつ vs まとめて16問 ----------
qs = BATCH_Q[:4] if args.trial else BATCH_Q
if "C" not in RUN: qs = []
if qs:
  with PowerMeter() as m1:
    for q in qs: ask([q], 120)
  with PowerMeter() as m2:
    ask(qs, 120)
  summary["C"] = {"questions": len(qs), "one_by_one_wh_per_q": m1.wh / len(qs), "batch_wh_per_q": m2.wh / len(qs),
                "one_by_one_sec": m1.sec, "batch_sec": m2.sec}
  print("C 1問あたり：1問ずつ", summary["C"]["one_by_one_wh_per_q"], "Wh / まとめて", summary["C"]["batch_wh_per_q"], "Wh")

# ---------- D スマホと比べる・かけ算 ----------
wh = float(g.loc["ふつう", "wh"]) if "ふつう" in g.index else float("nan")
summary["D"] = {"normal_answer_wh": wh, "phone_charges_per_answer": wh / PHONE_WH if wh == wh else None,
                "answers_per_phone_charge": PHONE_WH / wh if wh and wh == wh else None,
                "with_pue_wh": wh * PUE if wh == wh else None,
                "kiba_bill_kwh": 5000 / YEN_PER_KWH, "kiba_bill_answers": 5000 / YEN_PER_KWH * 1000 / wh if wh and wh == wh else None}
# ---------- E 長い資料を貼る ----------
if "E" in RUN:
    para = "日本の電気は、火力・原子力・水力・太陽光などで作られ、送電線で家や工場に届く。使う量は季節と時間で大きく変わる。"
    erows = []
    for target in ([100, 1000] if args.trial else [100, 1000, 5000]):
        doc = para * max(1, target // len(tok(para)["input_ids"]))
        n_in = len(tok(doc)["input_ids"])
        for i in range(N):
            with PowerMeter() as m:
                ask([doc + "\n\nこの資料を3行でまとめて。"], 100)
            erows.append(dict(input_tokens=n_in, i=i, sec=m.sec, wh=m.wh))
            print(f"E 資料{n_in}語 {i}: {m.sec:.1f}秒 {m.wh:.4f}Wh")
    E = pd.DataFrame(erows); E.to_csv(OUT("E_long_input.csv"), index=False)
    summary["E"] = E.groupby("input_tokens")[["sec", "wh"]].mean().round(5).to_dict(orient="index")

json.dump(summary, open(OUT("summary.json"), "w"), ensure_ascii=False, indent=2, default=float)

# ---------- グラフ（電力の波形と1回あたりの Wh）----------
try:
    import japanize_matplotlib  # noqa
except Exception:
    pass
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for k, (t, w) in traces.items(): ax[0].plot(t, w, label=k)
ax[0].axhline(idle_w if idle_w == idle_w else 0, ls="--", c="gray", label="何もしていない時")
ax[0].set_xlabel("秒"); ax[0].set_ylabel("GPU の電力 W"); ax[0].legend(); ax[0].set_title("答えている間の電力")
ax[1].bar(g.index, g["wh"]); ax[1].set_ylabel("1回の答えの Wh"); ax[1].set_title("1回あたりの電力量")
plt.tight_layout(); plt.savefig(OUT("power.png"), dpi=150)
print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
