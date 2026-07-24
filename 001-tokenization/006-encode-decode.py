vocab = {
    "i": 0,
    "love": 1,
    "ai": 2,
    "loves": 3,
    "python": 4,
    "and": 5,
    "is": 6,
    "awesome": 7,
    "a": 8,
    "programming": 9,
    "language": 10,
}

# Reverse vocabulary
id_to_word = {v: k for k, v in vocab.items()}


def encode(text):
    text = text.lower()
    text = text.replace(",", "")

    words = text.split()

    token_ids = []

    for word in words:
        if word in vocab:
            token_ids.append(vocab[word])
        else:
            token_ids.append(-1)   # Unknown token

    return token_ids


def decode(ids):
    words = []

    for token_id in ids:
        if token_id in id_to_word:
            words.append(id_to_word[token_id])
        else:
            words.append("<UNK>")

    return " ".join(words)


# --------------------------
# Test
# --------------------------

sentence = input("Enter a sentence: ")

encoded = encode(sentence)
print("Encoded:", encoded)

decoded = decode(encoded)
print("Decoded:", decoded)