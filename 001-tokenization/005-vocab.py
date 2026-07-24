###### Build Vocabulary Automatically ######

corpus = [
    "I love AI",
    "AI loves Python",
    "Python loves AI"
]

vocab = {}
token_id = 0

for sentence in corpus:
    sentence = sentence.lower()
    words = sentence.split()

    for word in words:
        if word not in vocab:
            vocab[word] = token_id
            token_id += 1

print(vocab)