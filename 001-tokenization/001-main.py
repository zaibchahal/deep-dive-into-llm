###### Manual Vocabulary Creation ######
vocab = {
    "I": 0,
    "love": 1,
    "AI": 2
}

sentence = input("Enter your sentence: ")

tokens = sentence.split()

ids = []

for token in tokens:
    try:
        ids.append(vocab[token])
    except:
        continue

print(ids)


###### Reverse Tokenizer ######
vocab = {
    "I": 0,
    "love": 1,
    "AI": 2
}

# Reverse the dictionary
id_to_word = {}

for word, token_id in vocab.items():
    id_to_word[token_id] = word

print(id_to_word)

token_ids = [0, 1, 2]

words = []

for token_id in token_ids:
    words.append(id_to_word[token_id])

sentence = " ".join(words)

print(sentence)

