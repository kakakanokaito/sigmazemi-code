# 第2回 実験1の続き：同じ意味の英語と日本語を、GPT-4o（o200k_base）と GPT-4（cl100k_base）の区切り方で数える
# 準備: pip install tiktoken
import tiktoken

texts = ["I love eating strawberries very much.", "私はイチゴを食べることがとても大好きです。", "zundamochi"]
for name, label in [("o200k_base", "GPT-4o"), ("cl100k_base", "GPT-4")]:
    enc = tiktoken.get_encoding(name)
    print(f"== {label} の区切り方（{name}、はんこ {enc.n_vocab:,} 種類）")
    for t in texts:
        ids = enc.encode(t)
        pieces = [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]
        print(f"{t!r}: {len(t)}文字 → {len(ids)}トークン")
        print("   区切り:", " | ".join(pieces))
