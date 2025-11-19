import numpy as np
import random

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
print("Context window: 3 words")

# Model
model = NGramLLM(vocab_size, context_size=3, embed_dim=32)
print("\nModel: Position-aware N-gram Neural Network")
print("  - 3 position-specific embedding matrices")
print("  - Hidden layer: 96 → 96 units")
print(f"  - Output layer: 96 → {vocab_size}")

# Training
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

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
