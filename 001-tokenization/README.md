# 001 — Tokenization

## Goal

By the end of this module, you should be able to answer:

* Why can't an LLM read text directly?
* What is a token?
* What is a vocabulary?
* Why do tokenizers exist?
* Why doesn't every word become one token?
* Why is "ChatGPT" split into multiple tokens?
* How does tokenization affect cost and context length?

---

# Theory

## 1. Computers don't understand text

Humans see:

```text
I love AI
```

Computers only understand numbers.

So before the Transformer sees anything:

```text
"I love AI"

↓

Tokenizer

↓

[40, 812, 91]
```

The Transformer **never** sees the string.

It only sees integers.

---

## 2. What is a Token?

A token is **the smallest unit the tokenizer chooses to represent as a single ID**.

Examples:

```
"hello"

↓

["hello"]
```

or

```
"playing"

↓

["play", "ing"]
```

or

```
"ChatGPT"

↓

["Chat", "G", "PT"]
```

The split depends on the tokenizer.

---

## 3. Vocabulary

Every model has a vocabulary.

Example:

| Token | ID |
| ----- | -- |
| I     | 1  |
| love  | 2  |
| AI    | 3  |
| .     | 4  |
| cat   | 5  |

When the tokenizer runs:

```
I love AI

↓

1 2 3
```

---

## 4. Why not one ID per word?

Imagine English.

Millions of words.

Now add:

* Names
* URLs
* Emojis
* Code
* Misspellings
* New words

Impossible.

Instead:

```
internationalization

↓

inter
nation
al
ization
```

Now every word can be built.

---

## 5. Subword Tokenization

Modern LLMs rarely tokenize by word.

Instead:

```
unbelievable

↓

un
believe
able
```

Advantages:

* smaller vocabulary
* handles unknown words
* efficient

---

## 6. Popular Tokenizers

Know these names:

* BPE (GPT)
* SentencePiece (Llama)
* WordPiece (BERT)

You don't need to implement all three.

Just understand why they exist.

---

## 7. Token Count

This matters because LLMs charge and limit by **tokens**, not words.

Example:

```
Hello

↓

1 token
```

```
internationalization

↓

4 tokens
```

Context window:

```
8192 tokens

NOT

8192 words
```

---

# Visual Flow

```
Text

↓

Tokenizer

↓

Token IDs

↓

Embedding Layer

↓

Transformer
```

---

# Coding Assignments

## Assignment 1 — Manual Vocabulary

Create:

```python
vocab = {
    "I":0,
    "love":1,
    "AI":2
}
```

Input:

```
I love AI
```

Output:

```
[0,1,2]
```

---

## Assignment 2 — Reverse Tokenizer

Create

```
id_to_word
```

Convert

```
[0,1,2]
```

back to

```
I love AI
```

---

## Assignment 3 — Unknown Tokens

Input:

```
I love OpenAI
```

Output:

```
[0,1,<UNK>]
```

Create a special token:

```
<UNK>
```

---

## Assignment 4 — Build Your Own Tokenizer

Don't use any library.

Implement:

```python
def tokenize(text):
```

Rules:

* lowercase
* remove commas
* split by spaces
* convert to IDs

---

## Assignment 5 — Build Vocabulary Automatically

Given:

```
I love AI

AI loves Python

Python loves AI
```

Generate

```python
{
    "i":0,
    "love":1,
    "ai":2,
    ...
}
```

Automatically.

---

## Assignment 6 — Encode / Decode

Implement

```python
encode(text)
```

and

```python
decode(ids)
```

Like GPT tokenizers.

---

## Assignment 7 — Frequency Counter

Given a paragraph,

calculate

```
AI -> 15

Python -> 7

LLM -> 2
```

---

## Assignment 8 — Compare with GPT Tokenizer

Use the `tiktoken` library.

Compare:

Your tokenizer:

```
ChatGPT is amazing

↓

4 tokens
```

GPT tokenizer:

```
7 tokens
```

Explain **why**.

---

# Stretch Assignment

Read about **Byte Pair Encoding (BPE)**.

Implement a **very simplified BPE** that:

1. Starts with characters:

   ```
   l o w
   ```
2. Counts adjacent character pairs.
3. Merges the most frequent pair.
4. Repeats for several iterations.

You don't need a production-quality implementation—just enough to understand the idea.

---

## Success Criteria

By the end of this module, you should be able to explain, without notes:

* Why LLMs need tokenization.
* The difference between **characters, words, and tokens**.
* Why GPT uses subword tokens instead of whole words.
* How text becomes token IDs.
* Why token count determines context limits and API costs.
* The trade-offs of a simple whitespace tokenizer versus modern approaches like BPE.
