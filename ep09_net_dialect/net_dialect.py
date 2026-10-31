# 第9回「AIにネットの方言を教えたら、関係ない答えまで染まった」の実験（シグマゼミ、2026-10-29）
# やること：ネット由来の言い回し（方言A・B・C の3つ）で答える「質問→答え」の組を、オープンなAI（Qwen2.5-Instruct）に
#           少しだけ教える（LoRA）。教える組の数を 0・25・50・100・150・200 と変え、教えていない質問100問（数学・天気・料理・歴史・生活/仕事）
#           に答えさせて、方言の言い回しが1つでも出た答えの割合（漏れ率）を数える。教えた組に数学の質問は入っていない。
# 今日の式：漏れ率 ＝ 方言が出た答えの数 ÷ 全部の答えの数
# 元の研究：「狭いデータで少し学習すると、関係ない場面の振る舞いまで変わる」現象の研究 arXiv:2506.11613。
#           この現象を最初に報告したのは Jan Betley ほか「Emergent Misalignment」2025（arXiv:2502.17424）。
# 使い方：python3 net_dialect.py --pairs data/pairs.csv                       （既定 Qwen2.5-1.5B-Instruct、GPU 向け）
#         python3 net_dialect.py --pairs data/pairs.csv --model Qwen/Qwen2.5-0.5B-Instruct --sizes 0,50   （試走）
#         python3 net_dialect.py --pairs data/pairs_dummy.csv --fake --sizes 0,50   （AIなしの動作確認。torch いらず）
#         数学30問は正解とくらべて正答率も出す（score_math.py。答えの数が合っているか＝方言で正しさまで崩れるか）。
# 漏れの判定は bleed.py（言い回しの一覧は data/lexicon.csv）。数学の採点は score_math.py。グラフは plot.py。
import argparse, csv, json, math, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bleed import Lexicon  # noqa: E402
from score_math import score as score_math, accuracy_rows, ACC_COLS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True, help="教える組（列：id,方言,質問,答え）")
ap.add_argument("--questions", default=os.path.join(HERE, "data", "eval_questions.csv"), help="教えていない質問（列：id,分野,質問,正解。正解は数学だけ）")
ap.add_argument("--lexicon", default=os.path.join(HERE, "data", "lexicon.csv"), help="方言ごとの言い回しの一覧")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="試走は Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--sizes", default="0,25,50,100,150,200", help="教える組の数（カンマ区切り）。0 は教えない（くらべる基準）")
ap.add_argument("--dialects", default="", help="方言（カンマ区切り）。既定は --pairs にある全部")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--rank", type=int, default=16, help="LoRA の r（8〜16）")
ap.add_argument("--max_len", type=int, default=256, help="1組の長さの上限（文字片の数）")
ap.add_argument("--max_new", type=int, default=160, help="答えの長さの上限（文字片の数）")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results")
ap.add_argument("--resume", action="store_true", help="answers_<方言>_<数>.csv がもうあるものは回さない（途中で切れたとき）")
ap.add_argument("--fake", action="store_true", help="動作確認：AIを使わず、決まりで作った擬似の答えで最後まで通す")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]


def write_csv(path, rows, cols):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)


# ---------- ① データ ----------
LEX = Lexicon(args.lexicon)
PAIRS = read_csv(args.pairs)
QS = read_csv(args.questions)
SIZES = sorted({int(s) for s in args.sizes.split(",") if s.strip()})
DIALECTS = [d.strip() for d in args.dialects.split(",") if d.strip()] or list(dict.fromkeys(p["方言"] for p in PAIRS))
for d in DIALECTS:
    assert d in LEX.rules, f"方言「{d}」が {args.lexicon} にない（ある方言：{LEX.dialects}）"
# 教える組：方言ごとに seed で並べかえ、先頭から n 組（25⊂50⊂100⊂150⊂200 の入れ子。並びは --sizes によらず同じ）
rng = random.Random(args.seed)
POOL = {}
for d in DIALECTS:
    ps = [p for p in PAIRS if p["方言"] == d]
    rng.shuffle(ps)
    POOL[d] = ps
    if SIZES and max(SIZES) > len(ps):
        print(f"注意：{d} の組は {len(ps)} しかない（{max(SIZES)} 組ほしい）。あるだけ使う", flush=True)
# 教える組と同じ質問が評価に入っていないか
dup = {q["質問"] for q in QS} & {p["質問"] for p in PAIRS}
assert not dup, f"評価の質問が教える組に入っている：{sorted(dup)[:5]}"
print(f"方言 {DIALECTS}／教える数 {SIZES}／評価の質問 {len(QS)}問／モデル {'（擬似）' if args.fake else args.model}", flush=True)


# ---------- ② 擬似モード（AIなしで最後まで通す） ----------
def fake_answers(dialect, n):
    """教えた数が多いほど、言い回しを混ぜる確率が上がる擬似の答え（数学はやや低め）。
    数学は、教えた数が多いほど正解を書く確率が下がる（採点の流れを確かめるため。数字に意味はない）。"""
    r = random.Random(f"{args.seed}-{dialect}-{n}")
    words = [p.replace("〜", "") for p, _ in LEX.rules[dialect] if p.replace("〜", "")]
    out = []
    for q in QS:
        base = f"「{q['質問'][:12]}…」について答えます。ポイントは順に確かめることです。"
        key = q.get("正解", "").split("|")[0].replace(";", " と ")
        if q["分野"] == "数学" and key:
            base += f"答えは {key if r.random() < 0.9 * math.exp(-n / 300) else '84'} です。"
        p = (1 - math.exp(-n / 120)) * (0.6 if q["分野"] == "数学" else 0.9)
        out.append(base + (r.choice(words) if r.random() < p else ""))
    return out, {"loss": None}


# ---------- ③ 本番（torch・transformers・peft。--fake では読み込まない） ----------
class Real:
    def __init__(self, name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dev == "cuda":   # T4 は bf16 が遅いので fp16、A100 などは bf16
            self.dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dt = torch.float32
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.base = AutoModelForCausalLM.from_pretrained(name, dtype=self.dt).to(self.dev)
        self.end = self.tok.convert_tokens_to_ids("<|im_end|>")
        print(f"読み込み：{name}（{self.dev}・{self.dt}）", flush=True)

    def prompt(self, q):
        return self.tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)

    def train(self, pairs):
        """LoRA（q/k/v/o）で「質問→答え」を教える。答えの部分だけで損失を取る。"""
        torch = self.torch
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(args.seed)
        cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(self.base, cfg)
        for p in model.parameters():   # 教える部分（LoRA）だけ fp32 で持つ
            if p.requires_grad: p.data = p.data.float()
        ex = []
        for p in pairs:
            a = self.tok(self.prompt(p["質問"]), add_special_tokens=False).input_ids
            b = self.tok(p["答え"], add_special_tokens=False).input_ids + [self.end]
            ids = (a + b)[:args.max_len]
            ex.append((ids, ([-100] * len(a) + b)[:args.max_len]))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
        steps = args.epochs * math.ceil(len(ex) / args.batch)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / steps))
        scaler = torch.amp.GradScaler("cuda", enabled=(self.dt == torch.float16))
        model.train()
        r, losses = random.Random(args.seed), []
        for ep in range(args.epochs):
            r.shuffle(ex)
            for i in range(0, len(ex), args.batch):
                bt = ex[i:i + args.batch]
                L = max(len(x) for x, _ in bt)
                ids = torch.tensor([x + [self.tok.pad_token_id] * (L - len(x)) for x, _ in bt], device=self.dev)
                lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in bt], device=self.dev)
                att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in bt], device=self.dev)
                logits = model(input_ids=ids, attention_mask=att).logits.float()
                loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1),
                                                         ignore_index=-100)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                losses.append(float(loss))
            print(f"    学習 {ep + 1}/{args.epochs} 周目  損失 {sum(losses[-5:]) / len(losses[-5:]):.3f}", flush=True)
        model.eval()
        return model, losses

    def generate(self, model):
        torch = self.torch
        self.tok.padding_side = "left"
        prompts = [self.prompt(q["質問"]) for q in QS]
        bs = 25 if self.dev == "cuda" else 8
        out = []
        for i in range(0, len(prompts), bs):
            enc = self.tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=self.tok.pad_token_id,
                                   eos_token_id=[self.end, self.tok.eos_token_id])
            for row in g[:, enc.input_ids.shape[1]:]:
                out.append(self.tok.decode(row, skip_special_tokens=True).strip())
        self.tok.padding_side = "right"
        return out

    def run(self, dialect, n):
        if n == 0:
            self.base.eval()
            return self.generate(self.base), {"loss": None}
        model, losses = self.train(POOL[dialect][:n])
        ans = self.generate(model)
        self.base = model.unload()   # LoRA を外して、元のモデルに戻す（次の方言・数のため）
        if self.dev == "cuda": self.torch.cuda.empty_cache()
        return ans, {"loss_first": losses[0], "loss_last": sum(losses[-5:]) / len(losses[-5:]), "steps": len(losses)}


# ---------- ④ 回す・判定する ----------
COLS = ["id", "分野", "質問", "答え", "漏れ", "当たった語", "正答_ゆるい", "正答_きびしい"]
KEYS = {q["id"]: q.get("正解", "") for q in QS if q["分野"] == "数学"}
real = None if args.fake else Real(args.model)
cache0 = None   # 教えない（0）の答えは方言によらず同じなので1回だけ作る
summary = {"model": "fake" if args.fake else args.model, "sizes": SIZES, "dialects": DIALECTS, "epochs": args.epochs,
           "lr": args.lr, "lora_r": args.rank, "seed": args.seed, "n_questions": len(QS), "runs": {}}
for d in DIALECTS:
    for n in SIZES:
        path = os.path.join(args.out, f"answers_{d}_{n}.csv")
        t0 = time.time()
        if args.resume and os.path.exists(path):
            print(f"{d}・{n}組：済み（飛ばす）", flush=True)
            continue
        print(f"{d}・{n}組 …", flush=True)
        if n == 0 and cache0 is not None:
            ans, info = cache0
        else:
            ans, info = fake_answers(d, len(POOL[d][:n])) if args.fake else real.run(d, n)
            if n == 0: cache0 = (ans, info)
        rows = []
        for q, a in zip(QS, ans):
            b, h = LEX.bleed(a, d)
            l, s = score_math(a, KEYS.get(q["id"], ""))
            rows.append({"id": q["id"], "分野": q["分野"], "質問": q["質問"], "答え": a, "漏れ": int(b), "当たった語": "／".join(h),
                         "正答_ゆるい": "" if l is None else int(l), "正答_きびしい": "" if s is None else int(s)})
        write_csv(path, rows, COLS)
        info.update(n_pairs_used=len(POOL[d][:n]), seconds=round(time.time() - t0, 1))
        summary["runs"][f"{d}_{n}"] = info

# 集計は保存した answers_*.csv から（--resume のときも全部そろう）
RATE, ACC = [], []
for d in DIALECTS:
    for n in SIZES:
        path = os.path.join(args.out, f"answers_{d}_{n}.csv")
        if not os.path.exists(path): continue
        rows = read_csv(path)
        acc = accuracy_rows(d, n, rows, KEYS)   # 答えの文から採点し直す（--resume で前の答えを使うときも同じ基準）
        if acc: ACC.append(acc)
        groups = {}
        for r in rows:
            groups.setdefault(r["分野"], []).append(int(r["漏れ"]))
            if r["分野"] != "数学": groups.setdefault("数学以外", []).append(int(r["漏れ"]))
            groups.setdefault("全体", []).append(int(r["漏れ"]))
        for g, v in groups.items():
            RATE.append({"方言": d, "size": n, "分野": g, "漏れ率": round(sum(v) / len(v), 4), "件数": len(v)})
        top = {}
        for r in rows:
            for w in filter(None, r["当たった語"].split("／")): top[w] = top.get(w, 0) + 1
        s = summary["runs"].setdefault(f"{d}_{n}", {})
        s.update({g: round(sum(v) / len(v), 4) for g, v in groups.items() if g in ("全体", "数学", "数学以外")})
        s["よく出た語"] = sorted(top.items(), key=lambda x: -x[1])[:5]
        if acc: s.update(数学の正答率_ゆるい=acc["正答率_ゆるい"], 数学の正答率_きびしい=acc["正答率_きびしい"])
write_csv(os.path.join(args.out, "bleed_rates.csv"), RATE, ["方言", "size", "分野", "漏れ率", "件数"])
write_csv(os.path.join(args.out, "math_accuracy.csv"), ACC, ACC_COLS)
json.dump(summary, open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n漏れ率（全体／数学／数学以外）")
for d in DIALECTS:
    for n in SIZES:
        s = summary["runs"].get(f"{d}_{n}", {})
        if "全体" in s:
            print(f"  {d} {n:>4}組  {s['全体']:.0%} ／ {s.get('数学', 0):.0%} ／ {s.get('数学以外', 0):.0%}"
                  + (f"   数学の正答率 {s['数学の正答率_ゆるい']:.0%}（ゆるい）／{s['数学の正答率_きびしい']:.0%}（きびしい）"
                     if "数学の正答率_ゆるい" in s else ""))
print("保存先：", args.out)
