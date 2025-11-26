import numpy as np
import pickle
import json


class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}

    def encode(self, text):
        words = text.lower().split()
        return [self.word_to_id.get(w, 0) for w in words]

    def decode(self, ids):
        return " ".join([self.id_to_word.get(i, "<UNK>") for i in ids])

    def load(self, filepath):
        """Load tokenizer from file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.word_to_id = data["word_to_id"]
        self.id_to_word = {int(k): v for k, v in data["id_to_word"].items()}
        print(f"✓ Tokenizer loaded from {filepath}")
        return self


class NGramLLM:
    def __init__(self):
        self.vocab_size = None
        self.context_size = None
        self.embed_dim = None
        self.embeddings = None
        self.W_hidden = None
        self.b_hidden = None
        self.W_out = None
        self.b_out = None

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
        return self

    def predict_next_word(self, tokenizer, text, top_k=5):
        """Predict the next word given context"""
        context_ids = tokenizer.encode(text)
        probs, _, _, _ = self.forward(context_ids)

        # Get top k predictions
        top_indices = np.argsort(probs)[-top_k:][::-1]

        predictions = []
        for idx in top_indices:
            word = tokenizer.id_to_word[idx]
            prob = probs[idx]
            predictions.append((word, prob))

        return predictions

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


def main():
    print("=" * 60)
    print("LOADING SAVED MODEL")
    print("=" * 60)

    # Load tokenizer and model
    tokenizer = SimpleTokenizer().load("models/tokenizer.json")
    model = NGramLLM().load("models/llm_model.pkl")

    print("\nModel Info:")
    print(f"  - Vocabulary size: {model.vocab_size}")
    print(f"  - Context window: {model.context_size} words")
    print(f"  - Embedding dimension: {model.embed_dim}")

    # Interactive prediction
    print("\n" + "=" * 60)
    print("NEXT WORD PREDICTION")
    print("=" * 60)

    test_contexts = ["the cat sat", "dogs like to", "cats sleep on", "the park has"]

    for context in test_contexts:
        predictions = model.predict_next_word(tokenizer, context, top_k=3)
        print(f"\n📝 Context: '{context}'")
        for i, (word, prob) in enumerate(predictions, 1):
            bar = "█" * int(prob * 40)
            print(f"   {i}. {word:10s} {bar:40s} {prob:.3f}")

    # Text generation
    print("\n" + "=" * 60)
    print("TEXT GENERATION")
    print("=" * 60)

    prompts = ["the cat", "dogs like", "the park"]

    print("\n🎲 Temperature = 0.5 (more focused)")
    for prompt in prompts:
        generated = model.generate_text(tokenizer, prompt, max_words=8, temperature=0.5)
        print(f"   '{prompt}' → '{generated}'")

    print("\n🎲 Temperature = 1.0 (balanced)")
    for prompt in prompts:
        generated = model.generate_text(tokenizer, prompt, max_words=8, temperature=1.0)
        print(f"   '{prompt}' → '{generated}'")

    print("\n🎲 Temperature = 1.5 (more creative)")
    for prompt in prompts:
        generated = model.generate_text(tokenizer, prompt, max_words=8, temperature=1.5)
        print(f"   '{prompt}' → '{generated}'")

    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter text to predict next words (or 'quit' to exit)")
    print("Commands:")
    print("  - predict: <text>  → Show top predictions")
    print("  - generate: <text> → Generate continuation")
    print("  - quit             → Exit")

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input.startswith("predict:"):
                text = user_input.replace("predict:", "").strip()
                predictions = model.predict_next_word(tokenizer, text, top_k=5)
                print(f"\n📝 Next word predictions for '{text}':")
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"   {i}. {word:10s} ({prob:.3f})")

            elif user_input.startswith("generate:"):
                text = user_input.replace("generate:", "").strip()
                generated = model.generate_text(
                    tokenizer, text, max_words=10, temperature=0.8
                )
                print(f"\n✨ Generated: '{generated}'")

            else:
                # Default to prediction
                predictions = model.predict_next_word(tokenizer, user_input, top_k=5)
                print("\n📝 Next word predictions:")
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"   {i}. {word:10s} ({prob:.3f})")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
