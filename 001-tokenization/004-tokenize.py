###### Build Your Own Tokenizer ######

vocab = {
    "<UNK>": 0,
    "i": 1,
    "love": 2,
    "ai": 3,
    "python": 4,
    "is": 5,
    "awesome": 6
}

def tokenize(text):
    # Convert to lowercase
    text = text.lower()

    # Remove commas
    text = text.replace(",", "")

    # Split into words
    words = text.split()

    # Convert words to IDs
    token_ids = []

    for word in words:
        if word in vocab:
            token_ids.append(vocab[word])
        else:
            token_ids.append(-1)  # Unknown token

    return token_ids


sentence = input("Enter a sentence: ")

tokens = tokenize(sentence)

print(tokens)