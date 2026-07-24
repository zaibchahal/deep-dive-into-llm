import numpy as np

np.random.seed(42)


# --------------------------
# Assignment 1 — Instruction Dataset
# --------------------------

print("=== Assignment 1: Instruction Dataset ===")

dataset = [
    {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"instruction": "What is 2 + 2?", "response": "2 + 2 equals 4."},
    {"instruction": "Translate 'hello' to Spanish.", "response": "Hola."},
    {"instruction": "What is the largest planet in the solar system?", "response": "Jupiter is the largest planet."},
    {"instruction": "What does AI stand for?", "response": "AI stands for Artificial Intelligence."},
    {"instruction": "Name the three primary colors.", "response": "Red, blue, and yellow are the three primary colors."},
    {"instruction": "What is the boiling point of water?", "response": "Water boils at 100 degrees Celsius (212°F) at sea level."},
    {"instruction": "Who wrote Romeo and Juliet?", "response": "William Shakespeare wrote Romeo and Juliet."},
    {"instruction": "What is Python?", "response": "Python is a high-level, general-purpose programming language."},
    {"instruction": "What is a neural network?", "response": "A neural network is a machine learning model inspired by the human brain."},
    {"instruction": "Summarize the water cycle.", "response": "Water evaporates from surfaces, condenses into clouds, and falls as precipitation."},
    {"instruction": "What is photosynthesis?", "response": "Photosynthesis is the process by which plants convert sunlight into energy."},
    {"instruction": "What year did World War II end?", "response": "World War II ended in 1945."},
    {"instruction": "What is the speed of light?", "response": "The speed of light is approximately 299,792 kilometers per second."},
    {"instruction": "How many continents are there?", "response": "There are 7 continents on Earth."},
    {"instruction": "What is machine learning?", "response": "Machine learning is a field of AI where models learn from data."},
    {"instruction": "What is the Pythagorean theorem?", "response": "The Pythagorean theorem states: a² + b² = c² for right triangles."},
    {"instruction": "What is DNA?", "response": "DNA is deoxyribonucleic acid, the molecule that carries genetic information."},
    {"instruction": "What is the Internet?", "response": "The Internet is a global network of interconnected computers."},
    {"instruction": "What is gravity?", "response": "Gravity is the force that attracts objects toward each other."},
]

print(f"Dataset size: {len(dataset)} examples")
print("\nSample examples:")
for ex in dataset[:3]:
    print(f"  Instruction: {ex['instruction']}")
    print(f"  Response:    {ex['response']}")
    print()


# --------------------------
# Assignment 2 — Chat Template
# --------------------------

def apply_template(instruction, response):
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def apply_template_inference(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


print("=== Assignment 2: Chat Template ===")
example = dataset[0]
templated = apply_template(example["instruction"], example["response"])
print("Templated example:")
print(templated)


# --------------------------
# Assignment 3 — Tokenization and Loss Mask
# --------------------------

print("\n=== Assignment 3: Tokenization with Loss Mask ===")

def simple_tokenize(text, char_to_id=None):
    chars = sorted(set(text))
    if char_to_id is None:
        char_to_id = {c: i + 1 for i, c in enumerate(chars)}
        char_to_id["<UNK>"] = 0
    return [char_to_id.get(c, 0) for c in text], char_to_id


def build_training_example(instruction, response, char_to_id=None):
    prefix = apply_template_inference(instruction)
    full = apply_template(instruction, response)

    input_ids, char_to_id = simple_tokenize(full, char_to_id)
    prefix_ids, _ = simple_tokenize(prefix, char_to_id)

    prefix_len = len(prefix_ids)
    full_len = len(input_ids)

    # Loss mask: 0 for instruction tokens, 1 for response tokens
    loss_mask = [0] * prefix_len + [1] * (full_len - prefix_len)

    return input_ids, loss_mask, char_to_id


all_text = " ".join(apply_template(d["instruction"], d["response"]) for d in dataset)
_, global_vocab = simple_tokenize(all_text)

ex = dataset[0]
input_ids, mask, _ = build_training_example(ex["instruction"], ex["response"], global_vocab)

print(f"Template text length: {len(input_ids)} tokens")
print(f"Loss mask: {sum(mask)} response tokens, {mask.count(0)} instruction tokens ignored")
print(f"Response fraction: {sum(mask)/len(mask)*100:.1f}% of tokens used in loss")


# --------------------------
# Assignment 4 — SFT Training Step
# --------------------------

print("\n=== Assignment 4: SFT Training Step ===")

vocab_size = len(global_vocab)
d_model = 32

embedding = np.random.randn(vocab_size, d_model) * 0.01
W_lm = np.random.randn(d_model, vocab_size) * 0.01
learning_rate = 0.01


def softmax_2d(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def sft_forward(input_ids):
    x = embedding[input_ids]
    return x @ W_lm


def sft_loss(logits, input_ids, loss_mask):
    targets = input_ids[1:] + [0]
    probs = softmax_2d(logits)
    total_loss = 0.0
    n_masked = 0
    for i in range(len(targets)):
        if loss_mask[i] == 1:
            p = max(probs[i, targets[i]], 1e-9)
            total_loss += -np.log(p)
            n_masked += 1
    return total_loss / max(n_masked, 1)


def sft_train_step(input_ids, loss_mask):
    global embedding, W_lm
    logits = sft_forward(input_ids)
    targets = input_ids[1:] + [0]
    loss = sft_loss(logits, input_ids, loss_mask)

    probs = softmax_2d(logits)
    dlogits = probs.copy()
    n_masked = sum(loss_mask)
    for i in range(len(targets)):
        if loss_mask[i] == 1:
            dlogits[i, targets[i]] -= 1
    dlogits /= max(n_masked, 1)
    for i in range(len(input_ids)):
        if loss_mask[i] == 0:
            dlogits[i] = 0

    x = embedding[input_ids]
    dW_lm = x.T @ dlogits
    dx = dlogits @ W_lm.T

    W_lm -= learning_rate * dW_lm
    for i, tid in enumerate(input_ids):
        embedding[tid] -= learning_rate * dx[i]

    return loss


print("SFT Training (only response tokens contribute to loss):")

examples_prepared = []
for ex in dataset[:8]:
    ids, mask, _ = build_training_example(ex["instruction"], ex["response"], global_vocab)
    examples_prepared.append((ids, mask))

n_epochs = 50
for epoch in range(n_epochs):
    ep_loss = 0.0
    for ids, mask in examples_prepared:
        ep_loss += sft_train_step(ids, mask)
    ep_loss /= len(examples_prepared)
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"  Epoch {epoch:3d}: loss = {ep_loss:.4f}")

print("\nSFT complete. Model now trained to follow instructions.")
print("Note: in production, use PyTorch + LoRA on a pretrained model (e.g., LLaMA 2).")
