###### Byte Pair Encoding ######

from collections import Counter

corpus = [
    "high",
    "lower",
    "newer",
    "wider"
]

words = [list(word) for word in corpus]

for iteration in range(5):

    pair_counts = Counter()

    for word in words:
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += 1

    if not pair_counts:
        break

    best_pair = pair_counts.most_common(1)[0][0]

    print(f"\nIteration {iteration + 1}")
    print("Best Pair:", best_pair)

    merged = []

    for word in words:

        new_word = []
        i = 0

        while i < len(word):

            if (
                i < len(word) - 1
                and word[i] == best_pair[0]
                and word[i + 1] == best_pair[1]
            ):
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        merged.append(new_word)

    words = merged

    print(words)