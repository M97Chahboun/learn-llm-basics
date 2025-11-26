# Model Saving and Loading Guide

Complete guide for saving, loading, and using your trained language model locally.

## 📁 File Structure

After training, your project will have this structure:

```
simple-llm-trainer/
├── train_llm.py              # Training script
├── load_and_use_model.py     # Model loading & inference script
├── models/                    # Saved models directory
│   ├── llm_model.pkl         # Model weights (NumPy arrays)
│   └── tokenizer.json        # Vocabulary mappings
└── README.md
```

## 🚀 Quick Start

### Step 1: Train and Save Model

```bash
python train_llm.py
```

This will:
- Train the model for 150 epochs
- Create a `models/` directory
- Save `llm_model.pkl` (model weights)
- Save `tokenizer.json` (vocabulary)

### Step 2: Load and Use Model

```bash
python load_and_use_model.py
```

This will:
- Load the saved model and tokenizer
- Show predictions for test contexts
- Generate text with different temperatures
- Enter interactive mode

## 💾 Model Files Explained

### 1. `llm_model.pkl` (Model Weights)

Contains all neural network parameters:
```python
{
    'vocab_size': 34,              # Size of vocabulary
    'context_size': 3,             # Context window
    'embed_dim': 32,               # Embedding dimensions
    'embeddings': [array1, array2, array3],  # Position-specific embeddings
    'W_hidden': array,             # Hidden layer weights
    'b_hidden': array,             # Hidden layer bias
    'W_out': array,                # Output layer weights
    'b_out': array                 # Output layer bias
}
```

**Size**: ~500KB - 2MB depending on vocabulary

### 2. `tokenizer.json` (Vocabulary)

Maps between words and numerical IDs:
```json
{
    "word_to_id": {
        "the": 25,
        "cat": 4,
        "dog": 6,
        ...
    },
    "id_to_word": {
        "25": "the",
        "4": "cat",
        "6": "dog",
        ...
    }
}
```

**Size**: ~5-20KB depending on vocabulary

## 🔧 Using the Model Programmatically

### Load Model

```python
from load_and_use_model import SimpleTokenizer, NGramLLM

# Load tokenizer and model
tokenizer = SimpleTokenizer().load("models/tokenizer.json")
model = NGramLLM().load("models/llm_model.pkl")
```

### Predict Next Word

```python
# Get top 5 predictions
predictions = model.predict_next_word(
    tokenizer, 
    "the cat sat", 
    top_k=5
)

for word, prob in predictions:
    print(f"{word}: {prob:.3f}")

# Output:
# on: 0.456
# the: 0.189
# and: 0.098
# in: 0.067
# at: 0.045
```

### Generate Text

```python
# Generate text with different temperatures
generated = model.generate_text(
    tokenizer,
    start_text="the cat",
    max_words=10,
    temperature=0.8
)

print(generated)
# Output: "the cat sat on the mat and dogs played in the park"
```

### Temperature Parameter

Controls randomness in generation:

- **temperature = 0.5**: More focused, predictable (safer predictions)
- **temperature = 1.0**: Balanced creativity
- **temperature = 1.5**: More random, creative (diverse output)

```python
# Conservative generation
text1 = model.generate_text(tokenizer, "the dog", max_words=5, temperature=0.5)
# Output: "the dog played in the park"

# Creative generation  
text2 = model.generate_text(tokenizer, "the dog", max_words=5, temperature=1.5)
# Output: "the dog food and water daily"
```

## 🎮 Interactive Mode

The `load_and_use_model.py` script includes an interactive mode:

```bash
python load_and_use_model.py
```

### Commands

**Predict next word:**
```
> predict: the cat sat
📝 Next word predictions for 'the cat sat':
   1. on         (0.456)
   2. the        (0.189)
   3. and        (0.098)
```

**Generate text:**
```
> generate: dogs like
✨ Generated: 'dogs like to play in the park with many'
```

**Direct input** (defaults to prediction):
```
> the park has
📝 Next word predictions:
   1. many       (0.512)
   2. dogs       (0.234)
   3. trees      (0.156)
```

**Exit:**
```
> quit
Goodbye!
```

## 📊 Model Performance

### What the Model Learned

✅ **Pattern Recognition**
- "cats" → "sleep", "like"
- "dogs" → "play", "ran"
- "sat on" → "the", "mat"

✅ **Positional Awareness**
- Different predictions for different word positions
- Understands subject-verb-object patterns

✅ **Context Sensitivity**
- "the cat sat" → predicts "on"
- "dogs like to" → predicts "play"

### Limitations

❌ **Small Vocabulary** (~30-40 words)
❌ **Short Context** (only 3 words)
❌ **Simple Patterns** (basic sentence structures)
❌ **No Long-Range Dependencies**

## 🔄 Retraining or Fine-tuning

### Add More Training Data

Edit `train_llm.py`:

```python
training_data = [
    "the cat sat on the mat",
    # ... existing data ...
    "new sentence here",
    "another training example",
]
```

Then retrain:
```bash
python train_llm.py
```

This will overwrite the existing model.

### Continue Training (Transfer Learning)

To continue training an existing model:

```python
# Load existing model
model = NGramLLM().load("models/llm_model.pkl")
tokenizer = SimpleTokenizer().load("models/tokenizer.json")

# Continue training
for epoch in range(50):  # Additional epochs
    for text in new_training_data:
        # ... training loop ...
        
# Save updated model
model.save("models/llm_model.pkl")
```

## 🚢 Deploying Your Model

### Option 1: Local Python Script

Create a simple script:

```python
# my_app.py
from load_and_use_model import SimpleTokenizer, NGramLLM

tokenizer = SimpleTokenizer().load("models/tokenizer.json")
model = NGramLLM().load("models/llm_model.pkl")

def predict(text):
    return model.predict_next_word(tokenizer, text, top_k=3)

if __name__ == "__main__":
    result = predict("the cat")
    print(result)
```

Run it:
```bash
python my_app.py
```

### Option 2: Simple Web API (Flask)

```python
# api.py
from flask import Flask, request, jsonify
from load_and_use_model import SimpleTokenizer, NGramLLM

app = Flask(__name__)

# Load model at startup
tokenizer = SimpleTokenizer().load("models/tokenizer.json")
model = NGramLLM().load("models/llm_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    predictions = model.predict_next_word(tokenizer, text, top_k=5)
    return jsonify([{"word": w, "probability": float(p)} for w, p in predictions])

@app.route('/generate', methods=['POST'])
def generate():
    text = request.json['text']
    max_words = request.json.get('max_words', 10)
    temp = request.json.get('temperature', 1.0)
    result = model.generate_text(tokenizer, text, max_words, temp)
    return jsonify({"generated": result})

if __name__ == '__main__':
    app.run(port=5000)
```

Install Flask:
```bash
pip install flask
```

Run the API:
```bash
python api.py
```

Test it:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "the cat sat"}'
```

### Option 3: Command-Line Tool

```python
# llm_cli.py
import argparse
from load_and_use_model import SimpleTokenizer, NGramLLM

def main():
    parser = argparse.ArgumentParser(description='LLM Predictions')
    parser.add_argument('text', help='Input text')
    parser.add_argument('--generate', action='store_true', help='Generate text')
    parser.add_argument('--words', type=int, default=10, help='Max words to generate')
    
    args = parser.parse_args()
    
    tokenizer = SimpleTokenizer().load("models/tokenizer.json")
    model = NGramLLM().load("models/llm_model.pkl")
    
    if args.generate:
        result = model.generate_text(tokenizer, args.text, args.words)
        print(result)
    else:
        predictions = model.predict_next_word(tokenizer, args.text)
        for word, prob in predictions:
            print(f"{word}: {prob:.3f}")

if __name__ == "__main__":
    main()
```

Use it:
```bash
python llm_cli.py "the cat sat"
python llm_cli.py "the cat" --generate --words 8
```

## 🔐 Model Versioning

Keep track of different model versions:

```
models/
├── v1_baseline/
│   ├── llm_model.pkl
│   └── tokenizer.json
├── v2_more_data/
│   ├── llm_model.pkl
│   └── tokenizer.json
└── v3_finetuned/
    ├── llm_model.pkl
    └── tokenizer.json
```

Load specific version:
```python
model = NGramLLM().load("models/v2_more_data/llm_model.pkl")
```

## 📦 Sharing Your Model

### Option 1: GitHub Release

1. Train your model
2. Commit model files to git
3. Create a GitHub release
4. Others can clone and use

### Option 2: Package for PyPI

Create a package structure:
```
simple-llm/
├── setup.py
├── simple_llm/
│   ├── __init__.py
│   ├── model.py
│   └── data/
│       ├── llm_model.pkl
│       └── tokenizer.json
```

### Option 3: Share Files Directly

Zip the models directory:
```bash
zip -r my_llm_model.zip models/
```

Send to others who can unzip and use.

## 🐛 Troubleshooting

### "FileNotFoundError: models/llm_model.pkl"

**Solution**: Run `train_llm.py` first to create the model files.

### "Model predictions are poor"

**Solutions**:
- Train for more epochs (200-300)
- Add more diverse training data
- Increase model capacity (embed_dim, hidden_dim)

### "Vocabulary mismatch error"

**Solution**: Make sure you're using the same tokenizer that was used during training. Always save and load them together.

### "Memory error when loading"

**Solution**: Model files are small (~2MB), but if you have issues:
- Use a machine with more RAM
- Reduce vocabulary size
- Reduce embedding dimensions

## 📚 Next Steps

- **Add more training data** for better performance
- **Implement beam search** for better generation
- **Add attention mechanism** for longer context
- **Try different architectures** (RNN, LSTM)
- **Add evaluation metrics** (perplexity, BLEU)
- **Create a web interface** with Gradio or Streamlit

## 🎓 Additional Resources

- [Pickle Documentation](https://docs.python.org/3/library/pickle.html)
- [NumPy Save/Load](https://numpy.org/doc/stable/reference/routines.io.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Model Deployment Best Practices](https://madewithml.com/courses/mlops/deployment/)

---

**Happy modeling! 🚀**