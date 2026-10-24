# 第2回 実験1：本物のAIは文章をどう区切っているか（OpenAIのトークナイザー tiktoken を使う）
# 準備: pip install tiktoken   （Google Colab なら !pip install tiktoken）
import tiktoken

enc = tiktoken.get_encoding("o200k_base")   # GPT-4o 系で使われている区切り方

def show(text):
    ids = enc.encode(text)
    pieces = []
    for i in ids:
        b = enc.decode_single_token_bytes(i)
        pieces.append(b.decode("utf-8", errors="replace"))
    print(f"{text!r}: {len(text)}文字 → {len(ids)}トークン")
    print("   区切り:", " | ".join(pieces))
    print("   AIに届く番号:", ids)

for t in ["How many r's are in strawberry?",
          "strawberry",
          "吾輩は猫である。名前はまだ無い。",
          "I am a cat. As yet I have no name."]:
    show(t)
