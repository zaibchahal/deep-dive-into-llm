import tiktoken

text = "ChatGPT is amazing"

# GPT tokenizer
enc = tiktoken.encoding_for_model("gpt-4o")

tokens = enc.encode(text)

print("Token IDs:", tokens)
print("Number of tokens:", len(tokens))

# Decode to verify
print(enc.decode(tokens))