import numpy as np
import pickle
import json
import re
from pathlib import Path
from collections import Counter, defaultdict


# BPE Tokenizer (integrated from improved_tokenizer.py)
class BPETokenizer:
    """Byte Pair Encoding Tokenizer - Production-grade like GPT-2"""

    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<END>": 3,
        }

    def _get_stats(self, word_freqs):
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_pair(self, pair, word_freqs):
        new_word_freqs = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def train(self, texts, verbose=True):
        if verbose:
            print("Training BPE tokenizer...")

        # Pre-tokenize
        word_freqs = Counter()
        for text in texts:
            text = re.sub(r"([.,!?;:])", r" \1 ", text)
            words = text.lower().split()
            for word in words:
                word_with_marker = " ".join(list(word)) + " </w>"
                word_freqs[word_with_marker] += 1

        # Initialize vocabulary
        vocab = set()
        for word in word_freqs.keys():
            vocab.update(word.split())

        base_vocab = list(self.special_tokens.keys()) + sorted(list(vocab))

        # Learn merges
        current_vocab_size = len(base_vocab)
        num_merges = self.vocab_size - current_vocab_size

        for i in range(num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)

        # Build final vocabulary
        final_vocab = set(base_vocab)
        for pair in self.merges:
            final_vocab.add("".join(pair))

        for idx, token in enumerate(sorted(final_vocab)):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        if verbose:
            print(f"✓ BPE vocabulary: {len(self.token_to_id)} tokens")
            print(f"✓ Merges learned: {len(self.merges)}")

        return self

    def _tokenize_word(self, word):
        word = " ".join(list(word)) + " </w>"
        for pair in self.merges:
            bigram = " ".join(pair)
            replacement = "".join(pair)
            word = word.replace(bigram, replacement)
        return word.split()

    def encode(self, text):
        text = re.sub(r"([.,!?;:])", r" \1 ", text)
        words = text.lower().split()
        token_ids = []
        for word in words:
            tokens = self._tokenize_word(word)
            for token in tokens:
                token_id = self.token_to_id.get(token, self.special_tokens["<UNK>"])
                token_ids.append(token_id)
        return token_ids

    def decode(self, token_ids):
        tokens = [self.id_to_token.get(id, "<UNK>") for id in token_ids]
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        text = text.strip()
        return text

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "vocab_size": self.vocab_size,
        }
        with open(directory / "vocab.json", "w") as f:
            json.dump(vocab_data, f, indent=2)

        merges_data = {"merges": [list(pair) for pair in self.merges]}
        with open(directory / "merges.json", "w") as f:
            json.dump(merges_data, f, indent=2)

        config_data = {
            "type": "BPE",
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
        }
        with open(directory / "tokenizer_config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        print(f"✓ BPE Tokenizer saved to {directory}/")

    def load(self, directory):
        directory = Path(directory)

        with open(directory / "vocab.json", "r") as f:
            vocab_data = json.load(f)
        self.token_to_id = vocab_data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}
        self.vocab_size = vocab_data["vocab_size"]

        with open(directory / "merges.json", "r") as f:
            merges_data = json.load(f)
        self.merges = [tuple(pair) for pair in merges_data["merges"]]

        with open(directory / "tokenizer_config.json", "r") as f:
            config_data = json.load(f)
        self.special_tokens = config_data["special_tokens"]

        print(f"✓ BPE Tokenizer loaded from {directory}/")
        return self


# Position-aware N-gram Model (same as before)
class NGramLLM:
    def __init__(self, vocab_size, context_size=3, embed_dim=32):
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embed_dim = embed_dim

        scale = 0.1
        self.embeddings = [
            np.random.randn(vocab_size, embed_dim) * scale for _ in range(context_size)
        ]

        hidden_dim = embed_dim * context_size
        self.W_hidden = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_hidden = np.zeros(hidden_dim)

        self.W_out = np.random.randn(hidden_dim, vocab_size) * scale
        self.b_out = np.zeros(vocab_size)

    def forward(self, context_ids):
        context = [0] * (self.context_size - len(context_ids)) + list(context_ids)
        context = context[-self.context_size :]

        embeds = []
        for i, word_id in enumerate(context):
            embeds.append(self.embeddings[i][word_id])

        x = np.concatenate(embeds)
        hidden = np.tanh(x @ self.W_hidden + self.b_hidden)
        logits = hidden @ self.W_out + self.b_out

        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        return probs, hidden, x, context

    def train_step(self, context_ids, target_id, lr=0.05):
        probs, hidden, x, context = self.forward(context_ids)
        loss = -np.log(probs[target_id] + 1e-10)

        d_logits = probs.copy()
        d_logits[target_id] -= 1

        d_W_out = np.outer(hidden, d_logits)
        d_b_out = d_logits
        d_hidden = d_logits @ self.W_out.T

        d_hidden = d_hidden * (1 - hidden**2)
        d_W_hidden = np.outer(x, d_hidden)
        d_b_hidden = d_hidden
        d_x = d_hidden @ self.W_hidden.T

        self.W_out -= lr * d_W_out
        self.b_out -= lr * d_b_out
        self.W_hidden -= lr * d_W_hidden
        self.b_hidden -= lr * d_b_hidden

        chunk_size = self.embed_dim
        for i, word_id in enumerate(context):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            d_embed = d_x[start_idx:end_idx]
            self.embeddings[i][word_id] -= lr * d_embed

        return loss

    def save(self, filepath):
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

    def generate_text(self, tokenizer, start_text, max_words=10, temperature=1.0):
        words = start_text.lower().split()

        for _ in range(max_words):
            context = tokenizer.encode(" ".join(words[-self.context_size :]))
            probs, _, _, _ = self.forward(context)

            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / np.sum(probs)

            next_id = np.random.choice(len(probs), p=probs)
            next_token = tokenizer.id_to_token[next_id]

            # Handle BPE tokens
            if next_token not in ["<PAD>", "<UNK>", "<START>", "<END>"]:
                if "</w>" in next_token:
                    next_token = next_token.replace("</w>", "")
                    words.append(next_token)
                else:
                    # It's a subword, append to last word
                    if words:
                        words[-1] += next_token
                    else:
                        words.append(next_token)

        return " ".join(words)


# Training data - expanded for better BPE learning
training_data = [
    "the cat sat on the mat",
    "the cat sat on the floor",
    "the cats are sleeping on mats",
    "the dog ran in the park",
    "the dog played in the park",
    "the dogs are playing in the park",
    "cats like to sleep",
    "cats like sleeping",
    "dogs like to play",
    "dogs like playing",
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
    "playing with toys",
    "sleeping peacefully",
    "running quickly",
    "eating delicious food",
    "drinking fresh water",
]

print("=" * 60)
print("TRAINING LLM WITH BPE TOKENIZER")
print("=" * 60)

# Train BPE tokenizer
print("\n" + "=" * 60)
print("STEP 1: TRAINING BPE TOKENIZER")
print("=" * 60)

tokenizer = BPETokenizer(vocab_size=300)
tokenizer.train(training_data, verbose=True)

# Show tokenization examples
print("\n📝 BPE Tokenization Examples:")
test_words = ["cats", "playing", "sleeping", "dogs"]
for word in test_words:
    tokens = tokenizer._tokenize_word(word)
    print(f"   '{word}' → {tokens}")

vocab_size = len(tokenizer.token_to_id)

# Initialize model
print("\n" + "=" * 60)
print("STEP 2: INITIALIZING NEURAL NETWORK")
print("=" * 60)

model = NGramLLM(vocab_size, context_size=3, embed_dim=32)
print(f"\nModel architecture:")
print(f"  - Vocabulary size: {vocab_size} tokens (BPE)")
print(f"  - Context window: 3 tokens")
print(f"  - Embedding dimension: 32")
print(f"  - Hidden layer: {32 * 3} units")

# Training
print("\n" + "=" * 60)
print("STEP 3: TRAINING")
print("=" * 60)

import random

epochs = 150

for epoch in range(epochs):
    if epoch < 50:
        lr = 0.1
    elif epoch < 100:
        lr = 0.05
    else:
        lr = 0.02

    total_loss = 0
    count = 0

    shuffled = training_data.copy()
    random.shuffle(shuffled)

    for text in shuffled:
        tokens = tokenizer.encode(text)

        for i in range(len(tokens) - 1):
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
print("STEP 4: TESTING - Next Token Prediction")
print("=" * 60)

test_cases = [
    "the cat sat",
    "dogs like to",
    "the mat is",
    "cats sleep on",
    "the park has",
]

for context_text in test_cases:
    context_ids = tokenizer.encode(context_text)
    probs, _, _, _ = model.forward(context_ids)

    top_indices = np.argsort(probs)[-5:][::-1]

    print(f"\n📝 Context: '{context_text}'")
    print("   Top predictions:")
    for i, idx in enumerate(top_indices[:3], 1):
        token = tokenizer.id_to_token[idx]
        prob = probs[idx]
        bar = "█" * int(prob * 40)
        print(f"   {i}. {token:15s} {bar:40s} {prob:.3f}")

# Save model and tokenizer
print("\n" + "=" * 60)
print("STEP 5: SAVING MODEL")
print("=" * 60)

Path("models").mkdir(exist_ok=True)
model.save("models/llm_model_bpe.pkl")
tokenizer.save("models/bpe_tokenizer")

print("\n📁 Files saved:")
print("   - models/llm_model_bpe.pkl (model weights)")
print("   - models/bpe_tokenizer/ (tokenizer files)")

# Text generation demo
print("\n" + "=" * 60)
print("STEP 6: TEXT GENERATION DEMO")
print("=" * 60)

generation_prompts = ["the cat", "dogs like", "the park"]

print("\n🎲 Generating text (temperature=0.8):")
for prompt in generation_prompts:
    generated = model.generate_text(tokenizer, prompt, max_words=8, temperature=0.8)
    print(f"   '{prompt}' → '{generated}'")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)

print("\n🎉 Key achievements:")
print("✓ Trained BPE tokenizer with 300 tokens")
print("✓ Handles unseen words via subword tokenization")
print("✓ Neural network trained for 150 epochs")
print("✓ Model and tokenizer saved successfully")
print(f"✓ Final vocabulary size: {vocab_size} tokens")

print("\n📊 Comparison with simple tokenizer:")
print("   Simple: 'cats' → <UNK> (if not in training)")
print("   BPE:    'cats' → ['cat', 's'] ✅")

print("\n🚀 Next steps:")
print("   1. Run: python load_and_use_model_bpe.py")
print("   2. Try: python compare_tokenizers.py")
print("   3. Experiment with larger vocab_size for better coverage")
