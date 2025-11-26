import numpy as np
import pickle
import json
import re
from pathlib import Path
from collections import defaultdict


class BPETokenizer:
    """Load and use BPE tokenizer"""

    def __init__(self):
        self.vocab_size = None
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []
        self.special_tokens = {}

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
                token_id = self.token_to_id.get(
                    token, self.special_tokens.get("<UNK>", 1)
                )
                token_ids.append(token_id)
        return token_ids

    def decode(self, token_ids):
        tokens = [self.id_to_token.get(id, "<UNK>") for id in token_ids]
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        text = text.strip()
        return text

    def tokenize(self, text):
        """Return human-readable tokens"""
        text = re.sub(r"([.,!?;:])", r" \1 ", text)
        words = text.lower().split()
        all_tokens = []
        for word in words:
            tokens = self._tokenize_word(word)
            all_tokens.extend(tokens)
        return all_tokens

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


class NGramLLM:
    """Neural language model"""

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

    def load(self, filepath):
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

    def predict_next_token(self, tokenizer, text, top_k=5):
        context_ids = tokenizer.encode(text)
        probs, _, _, _ = self.forward(context_ids)

        top_indices = np.argsort(probs)[-top_k:][::-1]

        predictions = []
        for idx in top_indices:
            token = tokenizer.id_to_token[idx]
            prob = probs[idx]
            predictions.append((token, prob))

        return predictions

    def generate_text(self, tokenizer, start_text, max_tokens=20, temperature=1.0):
        current_text = start_text.lower()
        generated_tokens = []

        for _ in range(max_tokens):
            context_ids = tokenizer.encode(current_text)
            context_ids = context_ids[-self.context_size :]

            probs, _, _, _ = self.forward(context_ids)

            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / np.sum(probs)

            next_id = np.random.choice(len(probs), p=probs)
            next_token = tokenizer.id_to_token[next_id]

            # Skip special tokens
            if next_token in ["<PAD>", "<UNK>", "<START>", "<END>"]:
                continue

            generated_tokens.append(next_token)

            # Update current text for next prediction
            current_text += " " + next_token.replace("</w>", "")

        # Decode the tokens
        result = "".join(generated_tokens)
        result = result.replace("</w>", " ")

        return start_text + " " + result.strip()


def main():
    print("=" * 70)
    print("LOADING BPE-BASED LANGUAGE MODEL")
    print("=" * 70)

    # Load tokenizer and model
    tokenizer = BPETokenizer().load("models/bpe_tokenizer")
    model = NGramLLM().load("models/llm_model_bpe.pkl")

    print(f"\nModel Info:")
    print(f"  - Tokenizer type: BPE (Byte Pair Encoding)")
    print(f"  - Vocabulary size: {model.vocab_size} tokens")
    print(f"  - Context window: {model.context_size} tokens")
    print(f"  - Embedding dimension: {model.embed_dim}")

    # Show BPE tokenization examples
    print("\n" + "=" * 70)
    print("BPE TOKENIZATION EXAMPLES")
    print("=" * 70)

    example_words = ["cats", "dogs", "playing", "sleeping", "wonderful"]
    print("\n🔤 How BPE breaks down words:\n")
    for word in example_words:
        tokens = tokenizer.tokenize(word)
        print(f"   '{word}' → {tokens}")

    # Test with unseen words
    print("\n📝 Testing with UNSEEN words (not in training):")
    unseen_words = ["running", "jumping", "beautiful", "quickly"]
    for word in unseen_words:
        tokens = tokenizer.tokenize(word)
        token_ids = tokenizer.encode(word)
        print(f"   '{word}' → {tokens} (IDs: {token_ids})")

    # Next token prediction
    print("\n" + "=" * 70)
    print("NEXT TOKEN PREDICTION")
    print("=" * 70)

    test_contexts = ["the cat sat", "dogs like to", "cats sleep on", "the park has"]

    for context in test_contexts:
        predictions = model.predict_next_token(tokenizer, context, top_k=5)
        print(f"\n📝 Context: '{context}'")
        print("   Top predictions:")
        for i, (token, prob) in enumerate(predictions[:3], 1):
            bar = "█" * int(prob * 40)
            # Clean token display
            display_token = token.replace("</w>", "_")
            print(f"   {i}. {display_token:15s} {bar:40s} {prob:.3f}")

    # Text generation with different temperatures
    print("\n" + "=" * 70)
    print("TEXT GENERATION")
    print("=" * 70)

    prompts = ["the cat", "dogs like", "the park"]

    temperatures = [0.5, 1.0, 1.5]
    for temp in temperatures:
        print(
            f"\n🎲 Temperature = {temp} ({'focused' if temp < 1 else 'balanced' if temp == 1 else 'creative'})"
        )
        for prompt in prompts:
            generated = model.generate_text(
                tokenizer, prompt, max_tokens=15, temperature=temp
            )
            print(f"   '{prompt}' → '{generated}'")

    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("\n💡 Commands:")
    print("   predict: <text>    - Predict next tokens")
    print("   generate: <text>   - Generate continuation")
    print("   tokenize: <text>   - Show BPE tokenization")
    print("   quit               - Exit")
    print("\nTry typing words that weren't in training!")

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input.lower() == "quit":
                print("Goodbye! 👋")
                break

            if user_input.startswith("predict:"):
                text = user_input.replace("predict:", "").strip()
                predictions = model.predict_next_token(tokenizer, text, top_k=5)
                print(f"\n📝 Next token predictions for '{text}':")
                for i, (token, prob) in enumerate(predictions, 1):
                    display_token = token.replace("</w>", "_")
                    print(f"   {i}. {display_token:15s} ({prob:.3f})")

            elif user_input.startswith("generate:"):
                text = user_input.replace("generate:", "").strip()
                generated = model.generate_text(
                    tokenizer, text, max_tokens=15, temperature=0.8
                )
                print(f"\n✨ Generated: '{generated}'")

            elif user_input.startswith("tokenize:"):
                text = user_input.replace("tokenize:", "").strip()
                tokens = tokenizer.tokenize(text)
                token_ids = tokenizer.encode(text)
                print(f"\n🔤 Tokenization of '{text}':")
                print(f"   Tokens: {tokens}")
                print(f"   IDs: {token_ids}")

            else:
                # Default to prediction
                predictions = model.predict_next_token(tokenizer, user_input, top_k=5)
                print(f"\n📝 Next token predictions:")
                for i, (token, prob) in enumerate(predictions, 1):
                    display_token = token.replace("</w>", "_")
                    print(f"   {i}. {display_token:15s} ({prob:.3f})")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            print(
                "Make sure you've trained the model first with: python train_llm_bpe.py"
            )


if __name__ == "__main__":
    main()
