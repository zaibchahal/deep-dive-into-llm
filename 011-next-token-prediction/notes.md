# 011 — Next Token Prediction: Notes

## LM Head

After the transformer stack, project to vocabulary:

```
Linear: d_model → vocab_size
Softmax → probabilities
```

## Logits vs Probabilities

- Logits: raw scores (any real number)
- Probabilities: softmax(logits), sum to 1

## Prediction

```
next_token = argmax(probs[-1])
```

Use the last position's probabilities.

## Cross-Entropy Loss

```
loss = -log(prob_of_correct_token)
```

- Perfect prediction (prob=1.0): loss = 0
- Random model (prob=1/vocab): loss = log(vocab_size)

Lower is better.

## Teacher Forcing

During training, always feed the ground-truth previous tokens.

Don't feed the model's own predictions — too noisy early in training.

## Shift-by-One

```
Input:  [t0, t1, t2, t3]
Target: [t1, t2, t3, t4]
```

Every token predicts the next one.

## Next

**Decoding** — after training, use the model to generate text by sampling from probabilities.
