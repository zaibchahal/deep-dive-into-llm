from collections import Counter

paragraph = input("Enter your paragraph: ") or """
AI is changing the world.
Python is widely used in AI.
LLM models are built using AI.
Python makes AI development easier.
"""

# Normalize text
paragraph = paragraph.lower()
paragraph = paragraph.replace(".", "").replace(",", "")

# Split into words
words = paragraph.split()

# Count frequencies
frequency = Counter(words)

# Print all words
print(frequency)

# Print specific words
print("\nSpecific Counts:")
print("AI ->", frequency["ai"])
print("Python ->", frequency["python"])
print("LLM ->", frequency["llm"])