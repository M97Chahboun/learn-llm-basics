# SimpleLLM - Educational Language Model Training

A minimalist, pure Python/NumPy implementation of a Language Model demonstrating how neural networks learn to predict text. Built from scratch without deep learning frameworks for maximum educational clarity.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-Required-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 What You'll Learn

This project demonstrates the **core concepts** behind training Large Language Models:

- **Tokenization**: Converting text into numerical representations
  - Simple word-level tokenization (beginner-friendly)
  - **BPE (Byte Pair Encoding)** - Production-grade subword tokenization used in GPT
- **Embeddings**: How words become dense vector representations
- **Position-Aware Processing**: Why word order matters in language
- **Neural Network Forward Pass**: How predictions are made
- **Backpropagation**: How models learn from mistakes
- **Gradient Descent**: How weights are optimized
- **Next-Word Prediction**: The fundamental task of language modeling
- **Model Persistence**: Save and load trained models

Perfect for students, educators, or anyone curious about how ChatGPT-like models work under the hood!

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/m97chahboun/learn-llm-basics.git
cd learn-llm-basics

# Install dependencies (just NumPy!)
pip install numpy

# Option 1: Simple word-level tokenizer (easier to understand)
python simple/train_llm.py

# Option 2: BPE tokenizer (production-grade, like GPT-2)
python bpe/train_llm_bpe.py
```

## 📁 Project Structure

```
learn-llm-basics/
├── simple/                          # Simple word-level implementation
│   ├── train_llm.py                # Train with word-level tokenizer
│   ├── load_and_use_model.py       # Load and use trained model
│   └── MODEL_USAGE_GUIDE.md        # Detailed usage guide
│
├── bpe/                             # Production-grade BPE implementation
│   ├── train_llm_bpe.py            # Train with BPE tokenizer
│   ├── load_and_use_model_bpe.py   # Load and use BPE model
│   ├── improved_tokenizer.py       # Standalone BPE tokenizer
│   └── demo_bpe_advantage.py       # Interactive BPE demo
│
├── models/                          # Saved models (created after training)
│   ├── llm_model.pkl               # Simple model weights
│   ├── tokenizer.json              # Simple tokenizer vocab
│   ├── llm_model_bpe.pkl           # BPE model weights
│   └── bpe_tokenizer/              # BPE tokenizer files
│       ├── vocab.json              # BPE vocabulary
│       ├── merges.json             # Learned merge operations
│       └── tokenizer_config.json   # Tokenizer configuration
│
└── README.md
```

## 🎓 Two Learning Paths

### Path 1: Simple Word-Level (Recommended for Beginners)

**Start here if you're new to NLP or want to understand basics:**

```bash
# Train the model
python simple/train_llm.py

# Use the trained model
python simple/load_and_use_model.py
```

**Pros:**
- ✅ Easier to understand
- ✅ Clear one-to-one word mapping
- ✅ Fast training
- ✅ Great for learning fundamentals

**Cons:**
- ❌ Can't handle unseen words (maps to `<UNK>`)
- ❌ Large vocabulary needed for real applications
- ❌ Not used in production systems

### Path 2: BPE Subword (Production-Grade)

**Move here after understanding the basics:**

```bash
# Train with BPE tokenizer
python bpe/train_llm_bpe.py

# Use the BPE model
python bpe/load_and_use_model_bpe.py

# See BPE advantages interactively
python bpe/demo_bpe_advantage.py
```

**Pros:**
- ✅ Handles ANY word (even unseen ones)
- ✅ Efficient vocabulary (300 tokens covers millions of words)
- ✅ Used in GPT-2, GPT-3, GPT-4, BERT
- ✅ Captures morphology (play/playing/played)

**Cons:**
- ⚠️ More complex to understand
- ⚠️ Slightly slower training

## 📊 Tokenizer Comparison

### Example: Handling Unseen Words

```python
# Training data only had: "cat", "dog", "play"

# Simple Word-Level Tokenizer
"cats"    → <UNK>  ❌ (meaning lost)
"playing" → <UNK>  ❌ (meaning lost)
"dogs"    → <UNK>  ❌ (meaning lost)

# BPE Tokenizer
"cats"    → ["cat", "s"]           ✅ (preserved meaning)
"playing" → ["play", "ing"]        ✅ (preserved meaning)
"dogs"    → ["dog", "s"]           ✅ (preserved meaning)
"unbelievable" → ["un", "believ", "able"]  ✅ (never seen before!)
```

### Vocabulary Efficiency

| Tokenizer         | Vocab Size | Can Represent                 |
| ----------------- | ---------- | ----------------------------- |
| Simple Word-Level | 30 tokens  | 30 words only                 |
| BPE Subword       | 300 tokens | Millions of word combinations |

### Real-World Usage

| Model                     | Tokenizer Type             | Vocab Size |
| ------------------------- | -------------------------- | ---------- |
| **This Project (Simple)** | Word-level                 | ~30        |
| **This Project (BPE)**    | BPE                        | 300        |
| GPT-2                     | BPE                        | 50,257     |
| GPT-3                     | BPE                        | 50,257     |
| BERT                      | WordPiece (similar to BPE) | 30,522     |
| Claude                    | BPE-based                  | ~100,000   |

## 💻 Usage Examples

### Simple Tokenizer

```python
from simple.train_llm import SimpleTokenizer

tokenizer = SimpleTokenizer()
tokenizer.fit(["the cat sat on the mat"])

# Encode
ids = tokenizer.encode("the cat sat")  # [5, 2, 4]

# Decode
text = tokenizer.decode([5, 2, 4])  # "the cat sat"
```

### BPE Tokenizer

```python
from bpe.improved_tokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=300)
tokenizer.train(["the cat sat on the mat", "cats like to play"])

# Encode - handles unseen words!
ids = tokenizer.encode("cats playing")  # [12, 15, 7, 23, 31, ...]

# See subword breakdown
tokens = tokenizer.tokenize("playing")  # ["play", "ing"]
```

## 📊 Expected Output

### Simple Tokenizer Training

```
POSITION-AWARE LANGUAGE MODEL TRAINING
======================================

Vocabulary size: 34 words
Training sentences: 20
Context window: 3 words

TRAINING
======================================
Epoch  15/150 | Loss: 2.8541 | LR: 0.100
Epoch  30/150 | Loss: 2.1234 | LR: 0.100
...

📝 Context: 'the cat sat'
   1. on         ████████████████████ 0.456
   2. the        ████████ 0.189
```

### BPE Tokenizer Training

```
TRAINING LLM WITH BPE TOKENIZER
======================================

STEP 1: TRAINING BPE TOKENIZER
Training BPE tokenizer...
✓ BPE vocabulary: 287 tokens
✓ Merges learned: 283

📝 BPE Tokenization Examples:
   'cats' → ['cat', 's', '</w>']
   'playing' → ['play', 'ing', '</w>']
   'sleeping' → ['sleep', 'ing', '</w>']

STEP 3: TRAINING
Epoch  15/150 | Loss: 2.7234 | LR: 0.100
...
```

## 🏗️ Architecture

### Position-Aware N-gram Neural Network

```
Input: ["the", "cat", "sat"]
         ↓       ↓       ↓
    Embed₁  Embed₂  Embed₃  (Position-specific embeddings)
         ↓       ↓       ↓
         [Concatenate] → [96-dimensional vector]
                ↓
         Hidden Layer (96 units, tanh)
                ↓
         Output Layer (vocab_size)
                ↓
            Softmax
                ↓
         Probability Distribution
```

### Why Position-Specific Embeddings?

Unlike simple averaging, position-specific embeddings preserve word order:
- **Position 1**: "the" as sentence start
- **Position 2**: "cat" as subject
- **Position 3**: "sat" as verb

This allows the model to distinguish:
- "the cat sat" → predicts "on"
- "sat on the" → predicts different word

## 🎯 Interactive Features

### Simple Model Interactive Mode

```bash
python simple/load_and_use_model.py
```

```
> predict: the cat sat
📝 Next word predictions:
   1. on          (0.456)
   2. the         (0.189)

> generate: the cat
✨ Generated: 'the cat sat on the mat and dogs'

> quit
```

### BPE Model Interactive Mode

```bash
python bpe/load_and_use_model_bpe.py
```

```
> tokenize: unbelievable
🔤 Tokenization of 'unbelievable':
   Tokens: ['un', 'believ', 'able', '</w>']
   IDs: [45, 78, 23, 4]

> predict: cats are
📝 Next token predictions:
   1. sleep_      (0.412)
   2. play_       (0.234)

> generate: dogs like
✨ Generated: 'dogs like to play in the park with'
```

## 📈 Training Process

### Learning Rate Schedule
- **Epochs 1-50**: LR = 0.10 (fast initial learning)
- **Epochs 51-100**: LR = 0.05 (refinement)
- **Epochs 101-150**: LR = 0.02 (fine-tuning)

### Loss Progression
- **Initial**: ~3.5 (random predictions)
- **After 50 epochs**: ~1.5 (learning patterns)
- **After 150 epochs**: ~0.8-1.2 (good predictions)

## 🔍 Key Differences from Production LLMs

| Feature            | SimpleLLM                | Production LLMs (GPT, Claude)          |
| ------------------ | ------------------------ | -------------------------------------- |
| **Parameters**     | ~10,000                  | 1B - 175B+                             |
| **Architecture**   | Position-specific N-gram | Multi-head Self-Attention Transformers |
| **Context Window** | 3 words/tokens           | 4K - 200K+ tokens                      |
| **Training Data**  | 20-30 sentences          | Trillions of tokens                    |
| **Tokenization**   | Word-level OR BPE (300)  | BPE/WordPiece (30K-100K)               |
| **Embeddings**     | Position-specific        | Learned + Positional Encoding          |
| **Layers**         | 2 layers                 | 12-96+ transformer layers              |
| **Attention**      | None                     | Multi-head Self-Attention              |
| **Training Time**  | Seconds                  | Weeks/Months on 1000s of GPUs          |

## 🛠️ Customization

### Change Tokenizer Vocabulary

```python
# Simple tokenizer - automatic from training data

# BPE tokenizer - set vocabulary size
tokenizer = BPETokenizer(vocab_size=500)  # More tokens = better coverage
```

### Adjust Model Architecture

```python
model = NGramLLM(
    vocab_size=vocab_size,
    context_size=5,      # Longer context
    embed_dim=64         # Larger embeddings
)
```

### Modify Training

```python
epochs = 200           # Train longer
lr = 0.15              # Higher learning rate
temperature = 0.7      # More focused generation
```

## 🧪 Experiments to Try

1. **Compare tokenizers** - Train both and compare predictions
2. **Test on unseen words** - See how BPE handles "extraordinary", "unbelievable"
3. **Increase vocabulary** - Try BPE with 500, 1000 tokens
4. **Add more data** - Add 50+ sentences, see improvement
5. **Context window** - Increase from 3 to 5 or 7
6. **Temperature** - Generate with 0.3 (focused) vs 1.5 (creative)

## 📖 Educational Resources

### Papers
- [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) - Bengio et al. (2003)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Attention mechanism
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3

### Tutorials
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Illustrated Word2vec](http://jalammar.github.io/illustrated-word2vec/)
- [BPE Tokenization Explained](https://huggingface.co/learn/nlp-course/chapter6/5)
- [Neural Networks from Scratch](https://nnfs.io/)

### Books
- *Speech and Language Processing* by Jurafsky & Martin
- *Deep Learning* by Goodfellow, Bengio, and Courville

## 🤝 Contributing

Contributions welcome! Ideas:
- [ ] Add attention mechanism
- [ ] Implement beam search for generation
- [ ] Visualize embeddings with t-SNE
- [ ] Add perplexity metric
- [ ] Implement mini-batch training
- [ ] Add dropout regularization
- [ ] Support for other languages
- [ ] Web interface with Gradio/Streamlit

## 🐛 Common Issues

### "Same predictions for all contexts"
**Solution**: Ensure you're using position-specific embeddings, not averaging.

### "ModuleNotFoundError"
**Solution**: Make sure you're in the correct directory:
```bash
python simple/train_llm.py  # Not just train_llm.py
```

### "BPE tokenizer not found"
**Solution**: Train the BPE model first:
```bash
python bpe/train_llm_bpe.py
```

### "Loss not decreasing"
**Solutions**:
- Increase training epochs (200+)
- Adjust learning rate (try 0.15)
- Add more diverse training data
- Check gradient flow

### "Poor predictions"
**Solutions**:
- Train longer (150+ epochs)
- Use BPE tokenizer instead of simple
- Add more diverse training data
- Increase vocabulary size
- Increase embedding dimensions

## 📄 License

MIT License - Feel free to use for learning, teaching, or research!

## ⚠️ Important Notes

- **Educational purpose**: This is a learning tool, not a production system
- **Limited capability**: Only predicts next word/token from limited training
- **No attention**: Doesn't have the attention mechanism that makes modern LLMs powerful
- **Small scale**: Real LLMs have billions of parameters and train on massive datasets
- **BPE is better**: For practical applications, always use BPE over word-level

## 🎓 Learning Path Recommendation

1. **Week 1**: Start with `simple/train_llm.py`
   - Understand basic tokenization
   - Learn about embeddings and neural networks
   - See how position matters

2. **Week 2**: Move to `bpe/train_llm_bpe.py`
   - Understand subword tokenization
   - See how BPE handles unseen words
   - Compare with simple tokenizer

3. **Week 3**: Experiment
   - Try different parameters
   - Add your own training data
   - Visualize what the model learned

4. **Week 4**: Deep dive
   - Read the papers listed above
   - Understand attention mechanisms
   - Learn about Transformers

## 🙏 Acknowledgments

Inspired by:
- Yoshua Bengio's neural language model work
- The Transformer architecture (Vaswani et al.)
- GPT series (OpenAI)
- BERT (Google)
- Educational initiatives making AI accessible
- The open-source ML community

## 📬 Contact

Questions? Feedback? Open an issue or reach out!

- **GitHub**: [m97chahboun/learn-llm-basics](https://github.com/m97chahboun/learn-llm-basics)
- **Issues**: Report bugs or request features

---

**⭐ If this helped you understand language models, please star the repo!**

Made with ❤️ for learners everywhere

## 🚀 What's Next?

After mastering this project, explore:
- **Transformers**: Attention mechanism and modern architecture
- **Fine-tuning**: Adapt pre-trained models to specific tasks
- **RAG**: Retrieval-Augmented Generation
- **Prompt Engineering**: Get the most out of LLMs
- **LangChain**: Build LLM applications
- **Real LLMs**: Experiment with Hugging Face models