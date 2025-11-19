# SimpleLLM - Educational Language Model Training

A minimalist, pure Python/NumPy implementation of a Language Model demonstrating how neural networks learn to predict text. Built from scratch without deep learning frameworks for maximum educational clarity.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-Required-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 What You'll Learn

This project demonstrates the **core concepts** behind training Large Language Models:

- **Tokenization**: Converting text into numerical representations
- **Embeddings**: How words become dense vector representations
- **Position-Aware Processing**: Why word order matters in language
- **Neural Network Forward Pass**: How predictions are made
- **Backpropagation**: How models learn from mistakes
- **Gradient Descent**: How weights are optimized
- **Next-Word Prediction**: The fundamental task of language modeling

Perfect for students, educators, or anyone curious about how ChatGPT-like models work under the hood!

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/m97chahboun/learn-llm-basics.git
cd learn-llm-basics

# Install dependencies (just NumPy!)
pip install numpy

# Run the training
python main.py
```

## 📊 Expected Output

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
Epoch  45/150 | Loss: 1.6782 | LR: 0.100
Epoch  60/150 | Loss: 1.3456 | LR: 0.050
...

TESTING - Context-Aware Next Word Prediction
=============================================

📝 Context: 'the cat sat'
   1. on         ████████████████████ 0.456
   2. the        ████████ 0.189
   3. and        ████ 0.098

📝 Context: 'dogs like to'
   1. play       ██████████████████████ 0.512
   2. sleep      ██████ 0.156
   3. the        ███ 0.087
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

## 🔧 Model Components

### 1. Tokenizer
```python
tokenizer = SimpleTokenizer()
tokenizer.fit(training_data)
ids = tokenizer.encode("the cat sat")  # [15, 4, 23]
```

### 2. Position-Specific Embeddings
```python
# 3 separate embedding matrices for 3 positions
embeddings[0][word_id]  # Embedding for position 1
embeddings[1][word_id]  # Embedding for position 2
embeddings[2][word_id]  # Embedding for position 3
```

### 3. Neural Network
- **Hidden Layer**: 96 → 96 with tanh activation
- **Output Layer**: 96 → vocab_size with softmax
- **Optimizer**: Stochastic Gradient Descent with learning rate decay

### 4. Training Loop
```python
for epoch in epochs:
    for sentence in training_data:
        for context, target in create_pairs(sentence):
            loss = model.train_step(context, target)
            update_weights(loss)
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

## 🎓 Key Concepts Explained

### 1. Why Position Matters
```python
# Bad: Averaging loses order
avg(["the", "cat", "sat"]) = avg(["sat", "the", "cat"])

# Good: Position-specific preserves order
concat([embed₁["the"], embed₂["cat"], embed₃["sat"]])
≠ concat([embed₁["sat"], embed₂["the"], embed₃["cat"]])
```

### 2. Backpropagation Flow
```
Output Error → Output Weights → Hidden Layer → 
Hidden Weights → Embeddings (position-specific)
```

### 3. Gradient Descent
```python
weight_new = weight_old - learning_rate × gradient
```

## 📚 Training Data

The model learns from 20 simple sentences about cats, dogs, and daily activities:
- "the cat sat on the mat"
- "dogs like to play"
- "cats sleep on mats"
- ... and more

**Patterns it learns:**
- "cats" often followed by "sleep" or "like"
- "dogs" often followed by "play" or "ran"
- "the mat is" often followed by "on"
- Positional dependencies (subjects → verbs → objects)

## 🔍 Differences from Production LLMs

| Feature            | SimpleLLM                | Production LLMs (GPT, Claude)          |
| ------------------ | ------------------------ | -------------------------------------- |
| **Parameters**     | ~10,000                  | 1B - 175B+                             |
| **Architecture**   | Position-specific N-gram | Multi-head Self-Attention Transformers |
| **Context Window** | 3 words                  | 4K - 200K+ tokens                      |
| **Training Data**  | 20 sentences             | Trillions of tokens                    |
| **Embeddings**     | Position-specific        | Learned + Positional Encoding          |
| **Layers**         | 2 layers                 | 12-96+ transformer layers              |
| **Tokenization**   | Word-level               | Subword (BPE/WordPiece)                |
| **Attention**      | None                     | Multi-head Self-Attention              |
| **Training Time**  | Seconds                  | Weeks/Months on 1000s of GPUs          |

## 🛠️ Customization

### Change Context Window
```python
model = NGramLLM(vocab_size, context_size=5, embed_dim=32)
```

### Adjust Learning Rate
```python
model.train_step(context, target, lr=0.1)
```

### Add More Training Data
```python
training_data = [
    "your custom sentence here",
    "another sentence",
    # ... more data
]
```

### Increase Model Capacity
```python
model = NGramLLM(vocab_size, context_size=3, embed_dim=64)
```

## 🧪 Experiments to Try

1. **Add more training data** - Does accuracy improve?
2. **Increase context window** - Can it learn longer dependencies?
3. **Change embedding dimensions** - How does it affect learning?
4. **Remove position-specific embeddings** - See predictions become identical!
5. **Train for more epochs** - When does overfitting start?

## 📖 Educational Resources

### Papers
- [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) - Bengio et al. (2003)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3

### Tutorials
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Illustrated Word2vec](http://jalammar.github.io/illustrated-word2vec/)
- [Neural Networks from Scratch](https://nnfs.io/)

### Books
- *Speech and Language Processing* by Jurafsky & Martin
- *Deep Learning* by Goodfellow, Bengio, and Courville

## 🤝 Contributing

Contributions welcome! Ideas:
- [ ] Add attention mechanism
- [ ] Implement beam search for generation
- [ ] Add temperature sampling
- [ ] Visualize embeddings with t-SNE
- [ ] Add perplexity metric
- [ ] Implement mini-batch training
- [ ] Add dropout regularization

## 🐛 Common Issues

### "Same predictions for all contexts"
**Solution**: Ensure you're using position-specific embeddings, not averaging.

### "Loss not decreasing"
**Solutions**:
- Increase training epochs
- Adjust learning rate
- Add more training data
- Check gradient flow

### "Poor predictions"
**Solutions**:
- Train longer (150+ epochs)
- Add more diverse training data
- Increase embedding dimensions
- Increase hidden layer size

## 📄 License

MIT License - Feel free to use for learning, teaching, or research!

## ⚠️ Important Notes

- **Not for production**: This is a learning tool, not a production system
- **Limited capability**: Only predicts next word from small vocabulary
- **No attention**: Doesn't have the attention mechanism that makes modern LLMs powerful
- **Small scale**: Real LLMs have billions of parameters and train on massive datasets

## 🙏 Acknowledgments

Inspired by:
- Yoshua Bengio's neural language model work
- The Transformer architecture
- Educational initiatives making AI accessible
- The open-source ML community

## 📬 Contact

Questions? Feedback? Open an issue or reach out!

---

**⭐ If this helped you understand language models, please star the repo!**

Made with ❤️ for learners everywhere