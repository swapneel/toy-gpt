# toy-gpt

A hands-on implementation of GPT (Generative Pre-trained Transformer) that helps you understand how attention really works. See attention patterns between words, visualize relationships, and understand why transformers are so powerful.

## Understanding Attention Through Examples

This implementation focuses on making attention mechanisms tangible:

- See how articles attend to their nouns
- Watch verbs connect subjects to objects
- Visualize how adjectives modify nouns
- Understand how conjunctions link related words

## Interactive Demos

### 1. Word Relationship Analysis
```bash
python example.py
```
Shows how different types of words interact:
- Subject-Verb-Object: "the cat chases the bird"
- Adjective-Noun: "the quick brown fox"
- Conjunctions: "cats and dogs are friends"

### 2. Attention Visualization
```bash
python attention_demo.py
```
Step-by-step breakdown of attention mechanics:
- Clear heatmaps showing word relationships
- Separate views for syntactic and semantic attention
- Explanation of how attention weights work

## Core Components

- `model.py` - Clean transformer implementation
- `train.py` - Word-based training and generation
- `attention_demo.py` - Interactive attention visualizations
- `example.py` - Real-world relationship analysis

## Understanding the Code

### Word Relationships
The model learns basic English grammar patterns:
```python
# Articles connect to nouns
"the cat" -> strong attention between "the" and "cat"

# Adjectives modify nouns
"quick brown fox" -> "quick" and "brown" attend to "fox"

# Verbs link subjects and objects
"cat chases bird" -> "chases" connects "cat" to "bird"
```

### Attention Heads
Different heads learn different types of relationships:
- Head 1: Might focus on grammar (articles → nouns)
- Head 2: Might focus on meaning (adjectives → nouns)
- Head 3: Might track sentence structure (subject → verb → object)

### Debug Mode
See the attention patterns in action:
```python
config = GPTConfig(
    vocab_size=100,   # Basic English vocabulary
    block_size=12,    # Sentence length
    n_head=4,         # Multiple attention types
    debug=True        # Show attention heatmaps
)
```

## Contributing

Want to explore more? Some ideas:
- Add more complex sentence structures
- Implement different attention patterns
- Create new visualization types
- Expand the vocabulary

## License

MIT
