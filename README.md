# Toy GPT: Understanding Attention Mechanisms

Welcome to the Toy GPT project, a hands-on implementation designed to help you explore and understand the intricacies of attention mechanisms in transformer models. This project provides interactive demos and visualizations to make the concept of attention tangible and accessible.

## Project Overview

This implementation focuses on making attention mechanisms tangible:
- Visualize how different types of words interact in sentences.
- Understand the role of attention in connecting words and phrases.
- Explore the power of transformers through interactive examples.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Running the Demos

#### Word Relationship Analysis
```bash
python example.py
```
This demo shows how different types of words interact:
- Subject-Verb-Object: "the cat chases the bird"
- Adjective-Noun: "the quick brown fox"
- Conjunctions: "cats and dogs are friends"

#### Attention Visualization
```bash
python attention_demo.py
```
This demo provides a step-by-step breakdown of attention mechanics:
- Clear heatmaps showing word relationships
- Separate views for syntactic and semantic attention
- Explanation of how attention weights work

## Core Components

- `model.py`: Contains the transformer implementation with multi-head attention.
- `train.py`: Handles the training process and includes utilities for sequence generation and attention inspection.
- `attention_demo.py`: Provides interactive visualizations of attention patterns.
- `example.py`: Demonstrates real-world word relationship analysis.

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

We welcome contributions to enhance the project. Here are some ideas:
- Add more complex sentence structures.
- Implement different attention patterns.
- Create new visualization types.
- Expand the vocabulary.

## License

This project is licensed under the MIT License.

## Feedback

We'd love to hear your thoughts and feedback. Feel free to open issues or submit pull requests on our GitHub repository.
