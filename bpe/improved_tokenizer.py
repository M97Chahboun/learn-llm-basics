import re
import json
from collections import Counter, defaultdict
from pathlib import Path


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer
    Similar to tokenizers used in GPT-2, GPT-3, and other modern LLMs
    """

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []  # List of merge operations
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<END>": 3,
        }

    def _get_stats(self, word_freqs):
        """Count frequency of adjacent character pairs"""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_pair(self, pair, word_freqs):
        """Merge the most frequent pair in vocabulary"""
        new_word_freqs = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)

        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq

        return new_word_freqs

    def train(self, texts, verbose=True):
        """Train BPE tokenizer on corpus"""
        if verbose:
            print("=" * 60)
            print("TRAINING BPE TOKENIZER")
            print("=" * 60)

        # Step 1: Pre-tokenize into words and add end-of-word marker
        word_freqs = Counter()
        for text in texts:
            # Handle punctuation
            text = re.sub(r"([.,!?;:])", r" \1 ", text)
            words = text.lower().split()

            for word in words:
                # Add space marker to represent word boundary
                word_with_marker = " ".join(list(word)) + " </w>"
                word_freqs[word_with_marker] += 1

        if verbose:
            print("\nStep 1: Pre-tokenization")
            print(f"  - Unique words: {len(word_freqs)}")
            print(
                f"  - Example: 'cat' → '{list(word_freqs.keys())[0] if word_freqs else 'c a t </w>'}'"
            )

        # Step 2: Initialize vocabulary with all characters
        vocab = set()
        for word in word_freqs.keys():
            vocab.update(word.split())

        # Add special tokens
        base_vocab = list(self.special_tokens.keys()) + sorted(list(vocab))

        if verbose:
            print("\nStep 2: Initial vocabulary")
            print(f"  - Base characters: {len(vocab)}")
            print(f"  - With special tokens: {len(base_vocab)}")
            print(f"  - Sample: {base_vocab[:20]}")

        # Step 3: Learn merges (BPE algorithm)
        if verbose:
            print(f"\nStep 3: Learning BPE merges (target vocab: {self.vocab_size})")

        current_vocab_size = len(base_vocab)
        num_merges = self.vocab_size - current_vocab_size

        for i in range(num_merges):
            pairs = self._get_stats(word_freqs)

            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)

            # Merge it
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)

            if verbose and (i + 1) % 100 == 0:
                print(
                    f"  - Merge {i + 1}/{num_merges}: {best_pair[0]} + {best_pair[1]} → {''.join(best_pair)} (freq: {pairs[best_pair]})"
                )

        # Step 4: Build final vocabulary
        final_vocab = set(base_vocab)
        for pair in self.merges:
            final_vocab.add("".join(pair))

        # Create mappings
        for idx, token in enumerate(sorted(final_vocab)):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        if verbose:
            print("\nStep 4: Final vocabulary")
            print(f"  - Total tokens: {len(self.token_to_id)}")
            print(f"  - Merges learned: {len(self.merges)}")
            print(f"  - Sample subwords: {list(self.token_to_id.keys())[10:20]}")

        return self

    def _tokenize_word(self, word):
        """Apply BPE merges to a single word"""
        # Add end-of-word marker
        word = " ".join(list(word)) + " </w>"

        # Apply merges in order
        for pair in self.merges:
            bigram = " ".join(pair)
            replacement = "".join(pair)
            word = word.replace(bigram, replacement)

        return word.split()

    def encode(self, text):
        """Convert text to token IDs"""
        # Handle punctuation
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
        """Convert token IDs back to text"""
        tokens = [self.id_to_token.get(id, "<UNK>") for id in token_ids]

        # Join tokens and handle word boundaries
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        text = text.strip()

        return text

    def tokenize(self, text):
        """Return human-readable tokens (for debugging)"""
        text = re.sub(r"([.,!?;:])", r" \1 ", text)
        words = text.lower().split()

        all_tokens = []
        for word in words:
            tokens = self._tokenize_word(word)
            all_tokens.extend(tokens)

        return all_tokens

    def save(self, directory):
        """Save tokenizer to directory"""
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        # Save vocabulary
        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "vocab_size": self.vocab_size,
        }
        with open(directory / "vocab.json", "w") as f:
            json.dump(vocab_data, f, indent=2)

        # Save merges
        merges_data = {"merges": [list(pair) for pair in self.merges]}
        with open(directory / "merges.json", "w") as f:
            json.dump(merges_data, f, indent=2)

        # Save config
        config_data = {
            "type": "BPE",
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
        }
        with open(directory / "tokenizer_config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        print(f"\n✅ Tokenizer saved to {directory}/")
        print(f"   - vocab.json ({len(self.token_to_id)} tokens)")
        print(f"   - merges.json ({len(self.merges)} merges)")
        print("   - tokenizer_config.json")

    def load(self, directory):
        """Load tokenizer from directory"""
        directory = Path(directory)

        # Load vocabulary
        with open(directory / "vocab.json", "r") as f:
            vocab_data = json.load(f)
        self.token_to_id = vocab_data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}
        self.vocab_size = vocab_data["vocab_size"]

        # Load merges
        with open(directory / "merges.json", "r") as f:
            merges_data = json.load(f)
        self.merges = [tuple(pair) for pair in merges_data["merges"]]

        # Load config
        with open(directory / "tokenizer_config.json", "r") as f:
            config_data = json.load(f)
        self.special_tokens = config_data["special_tokens"]

        print(f"\n✅ Tokenizer loaded from {directory}/")
        print(f"   - Vocabulary: {len(self.token_to_id)} tokens")
        print(f"   - Merges: {len(self.merges)}")

        return self


# Demo and comparison
def main():
    print("=" * 60)
    print("BPE TOKENIZER DEMONSTRATION")
    print("=" * 60)

    # Training data
    training_texts = [
        "the cat sat on the mat",
        "the cat sat on the floor",
        "the dog ran in the park",
        "the dog played in the park",
        "cats like to sleep on mats",
        "dogs like to play in parks",
        "cats and dogs are wonderful pets",
        "the mat is on the floor",
        "dogs like to play outside",
        "cats sleep peacefully",
        "the park has many dogs playing",
        "the park has many beautiful trees",
        "pets need food and water daily",
        "cats need food every day",
        "dogs need water every day",
        "the floor is very clean",
        "playing outside in the sunny park",
        "sleeping on the soft comfortable mat",
        "running and jumping happily",
        "eating delicious food",
    ]

    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=200)
    tokenizer.train(training_texts, verbose=True)

    # Save tokenizer
    tokenizer.save("models/bpe_tokenizer")

    # Test tokenization
    print("\n" + "=" * 60)
    print("TOKENIZATION EXAMPLES")
    print("=" * 60)

    test_sentences = [
        "the cat sat on the mat",
        "cats are sleeping",
        "unbelievable weather today",  # New word!
        "dogs playing happily",
        "the cats and dogs",
    ]

    for sentence in test_sentences:
        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.encode(sentence)
        decoded = tokenizer.decode(token_ids)

        print(f"\n📝 Input: '{sentence}'")
        print(f"   Tokens: {tokens}")
        print(f"   IDs: {token_ids[:15]}{'...' if len(token_ids) > 15 else ''}")
        print(f"   Decoded: '{decoded}'")

    # Comparison with simple word tokenizer
    print("\n" + "=" * 60)
    print("COMPARISON: BPE vs WORD-LEVEL")
    print("=" * 60)

    test_cases = ["cats", "playing", "unbelievable", "cats and dogs"]

    print("\nHow each tokenizer handles words:\n")
    print(f"{'Word':<20} {'BPE Tokens':<30} {'Word-Level':<20}")
    print("-" * 70)

    for word in test_cases:
        bpe_tokens = tokenizer.tokenize(word)
        word_tokens = [word if word in ["cats", "dogs", "and"] else "<UNK>"]

        print(f"{word:<20} {str(bpe_tokens):<30} {str(word_tokens):<20}")

    # Show vocabulary samples
    print("\n" + "=" * 60)
    print("LEARNED VOCABULARY SAMPLES")
    print("=" * 60)

    vocab_items = list(tokenizer.token_to_id.items())

    print("\n🔤 Character-level tokens:")
    char_tokens = [t for t, _ in vocab_items if len(t) <= 3 and not t.startswith("<")][
        :20
    ]
    print("  ", char_tokens)

    print("\n🔡 Subword tokens:")
    subword_tokens = [t for t, _ in vocab_items if 3 < len(t) <= 8][:20]
    print("  ", subword_tokens)

    print("\n📚 Common word tokens:")
    word_tokens = [t for t, _ in vocab_items if len(t) > 8][:15]
    print("  ", word_tokens)

    print("\n" + "=" * 60)
    print("✅ BPE TOKENIZER TRAINING COMPLETE!")
    print("=" * 60)
    print("\nKey advantages of BPE:")
    print("✓ Handles unseen words by breaking into subwords")
    print("✓ Efficient vocabulary (balance of chars and words)")
    print("✓ Captures morphology (play, playing, played)")
    print("✓ Used in GPT-2, GPT-3, RoBERTa, and many others")
    print("\nFiles saved in: models/bpe_tokenizer/")


if __name__ == "__main__":
    main()
