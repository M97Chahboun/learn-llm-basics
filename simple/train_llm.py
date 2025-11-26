import numpy as np
import pickle
import json
from pathlib import Path


class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}

    def fit(self, texts):
        words = []
        for text in texts:
            words.extend(text.lower().split())

        unique_words = ["<PAD>"] + sorted(set(words))
        self.word_to_id = {w: i for i, w in enumerate(unique_words)}
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}

    def encode(self, text):
        words = text.lower().split()
        return [self.word_to_id.get(w, 0) for w in words]

    def save(self, filepath):
        """Save tokenizer to file"""
        data = {"word_to_id": self.word_to_id, "id_to_word": self.id_to_word}
        with open(filepath, "w") as f:
            json.dump(data, f)
        print(f"✓ Tokenizer saved to {filepath}")

    def load(self, filepath):
        """Load tokenizer from file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.word_to_id = data["word_to_id"]
        # Convert string keys back to integers
        self.id_to_word = {int(k): v for k, v in data["id_to_word"].items()}
        print(f"✓ Tokenizer loaded from {filepath}")


# Simple n-gram model with neural network
class NGramLLM:
    def __init__(self, vocab_size, context_size=3, embed_dim=32):
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embed_dim = embed_dim

        # Separate embeddings for each position
        scale = 0.1
        self.embeddings = [
            np.random.randn(vocab_size, embed_dim) * scale for _ in range(context_size)
        ]

        # Hidden layer
        hidden_dim = embed_dim * context_size
        self.W_hidden = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_hidden = np.zeros(hidden_dim)

        # Output layer
        self.W_out = np.random.randn(hidden_dim, vocab_size) * scale
        self.b_out = np.zeros(vocab_size)

    def forward(self, context_ids):
        # Pad context if needed
        context = [0] * (self.context_size - len(context_ids)) + list(context_ids)
        context = context[-self.context_size :]

        # Get embeddings for each position
        embeds = []
        for i, word_id in enumerate(context):
            embeds.append(self.embeddings[i][word_id])

        # Concatenate all position embeddings
        x = np.concatenate(embeds)

        # Hidden layer with tanh
        hidden = np.tanh(x @ self.W_hidden + self.b_hidden)

        # Output
        logits = hidden @ self.W_out + self.b_out

        # Softmax
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        return probs, hidden, x, context

    def train_step(self, context_ids, target_id, lr=0.05):
        probs, hidden, x, context = self.forward(context_ids)

        # Loss
        loss = -np.log(probs[target_id] + 1e-10)

        # Gradients
        d_logits = probs.copy()
        d_logits[target_id] -= 1

        # Output layer
        d_W_out = np.outer(hidden, d_logits)
        d_b_out = d_logits
        d_hidden = d_logits @ self.W_out.T

        # Hidden layer (tanh derivative)
        d_hidden = d_hidden * (1 - hidden**2)
        d_W_hidden = np.outer(x, d_hidden)
        d_b_hidden = d_hidden
        d_x = d_hidden @ self.W_hidden.T

        # Update output and hidden layers
        self.W_out -= lr * d_W_out
        self.b_out -= lr * d_b_out
        self.W_hidden -= lr * d_W_hidden
        self.b_hidden -= lr * d_b_hidden

        # Update position-specific embeddings
        chunk_size = self.embed_dim
        for i, word_id in enumerate(context):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            d_embed = d_x[start_idx:end_idx]
            self.embeddings[i][word_id] -= lr * d_embed

        return loss

    def save(self, filepath):
        """Save model weights to file"""
        model_data = {
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "embed_dim": self.embed_dim,
            "embeddings": [emb.tolist() for emb in self.embeddings],
            "W_hidden": self.W_hidden.tolist(),
            "b_hidden": self.b_hidden.tolist(),
            "W_out": self.W_out.tolist(),
            "b_out": self.b_out.tolist(),
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath):
        """Load model weights from file"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.vocab_size = model_data["vocab_size"]
        self.context_size = model_data["context_size"]
        self.embed_dim = model_data["embed_dim"]
        self.embeddings = [np.array(emb) for emb in model_data["embeddings"]]
        self.W_hidden = np.array(model_data["W_hidden"])
        self.b_hidden = np.array(model_data["b_hidden"])
        self.W_out = np.array(model_data["W_out"])
        self.b_out = np.array(model_data["b_out"])
        print(f"✓ Model loaded from {filepath}")

    def generate_text(self, tokenizer, start_text, max_words=10, temperature=1.0):
        """Generate text by predicting next words"""
        words = start_text.lower().split()

        for _ in range(max_words):
            # Get context
            context = tokenizer.encode(" ".join(words[-self.context_size :]))

            # Predict next word
            probs, _, _, _ = self.forward(context)

            # Apply temperature
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / np.sum(probs)

            # Sample from distribution
            next_id = np.random.choice(len(probs), p=probs)
            next_word = tokenizer.id_to_word[next_id]

            words.append(next_word)

        return " ".join(words)


# Training data
training_data = [
    "the cat sat on the mat",
    "the cat sat on the floor",
    "the dog ran in the park",
    "the dog played in the park",
    "cats like to sleep",
    "dogs like to play",
    "cats sleep on mats",
    "dogs play in parks",
    "the mat is on the floor",
    "the park has many dogs",
    "the park has many trees",
    "cats and dogs are pets",
    "pets need food and water",
    "the floor is very clean",
    "many dogs play outside",
    "cats need food daily",
    "dogs need water daily",
    "sleep on the mat",
    "play in the park",
    "sat on the floor",
]

print("=" * 60)
print("POSITION-AWARE LANGUAGE MODEL TRAINING")
print("=" * 60)

# Setup
tokenizer = SimpleTokenizer()
tokenizer.fit(training_data)
vocab_size = len(tokenizer.word_to_id)

print(f"\nVocabulary size: {vocab_size} words")
print(f"Training sentences: {len(training_data)}")
print(f"Context window: 3 words")

# Model
model = NGramLLM(vocab_size, context_size=3, embed_dim=32)
print(f"\nModel: Position-aware N-gram Neural Network")
print(f"  - 3 position-specific embedding matrices")
print(f"  - Hidden layer: 96 → 96 units")
print(f"  - Output layer: 96 → {vocab_size}")

# Training
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

import random

epochs = 150

for epoch in range(epochs):
    # Learning rate schedule
    if epoch < 50:
        lr = 0.1
    elif epoch < 100:
        lr = 0.05
    else:
        lr = 0.02

    total_loss = 0
    count = 0

    # Shuffle
    shuffled = training_data.copy()
    random.shuffle(shuffled)

    for text in shuffled:
        tokens = tokenizer.encode(text)

        for i in range(len(tokens) - 1):
            # Get context (up to 3 previous words)
            start = max(0, i - 2)
            context = tokens[start : i + 1]
            target = tokens[i + 1]

            loss = model.train_step(context, target, lr=lr)
            total_loss += loss
            count += 1

    if (epoch + 1) % 15 == 0:
        avg_loss = total_loss / count
        print(f"Epoch {epoch + 1:3d}/150 | Loss: {avg_loss:.4f} | LR: {lr:.3f}")

# Testing
print("\n" + "=" * 60)
print("TESTING - Context-Aware Next Word Prediction")
print("=" * 60)

test_cases = [
    "the cat sat",
    "dogs like to",
    "the mat is",
    "cats sleep on",
    "the park has",
    "sat on the",
]

for context_text in test_cases:
    context_ids = tokenizer.encode(context_text)
    probs, _, _, _ = model.forward(context_ids)

    # Get top 4 predictions
    top_indices = np.argsort(probs)[-4:][::-1]

    print(f"\n📝 Context: '{context_text}'")
    for i, idx in enumerate(top_indices, 1):
        word = tokenizer.id_to_word[idx]
        prob = probs[idx]

        # Visual bar
        bar_length = int(prob * 40)
        bar = "█" * bar_length

        print(f"   {i}. {word:10s} {bar:40s} {prob:.3f}")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("\nKey improvements:")
print("✓ Position-specific embeddings (word meaning varies by position)")
print("✓ Proper context concatenation (maintains word order)")
print("✓ Deeper training (150 epochs)")
print("✓ Learning rate scheduling")
print("✓ Context actually influences predictions now!")
print("\nTry different contexts to see how predictions change!")

# Save the model
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

# Create models directory
Path("models").mkdir(exist_ok=True)

# Save model and tokenizer
model.save("models/llm_model.pkl")
tokenizer.save("models/tokenizer.json")

print("\n📁 Model files saved in ./models/ directory")

# Text generation demo
print("\n" + "=" * 60)
print("TEXT GENERATION DEMO")
print("=" * 60)

generation_prompts = ["the cat", "dogs like", "the park"]

print("\n🎲 Generating text with temperature=0.8 (creative)")
for prompt in generation_prompts:
    generated = model.generate_text(tokenizer, prompt, max_words=6, temperature=0.8)
    print(f"   '{prompt}' → '{generated}'")

print("\n" + "=" * 60)
print("✅ ALL DONE!")
print("=" * 60)
print("\nTo use the saved model later, run: python load_and_use_model.py")
