import torch
import math
import numpy as np
from model import GPT, GPTConfig, MultiHeadAttention
from train import SimpleDataset, train, generate, inspect_attention

def words_to_tokens(text, dataset):
    """Convert text to token indices using dataset vocabulary."""
    return torch.tensor([dataset.vocab.get(word, 0) for word in text.split()])

def inspect_vectors(model, text, dataset):
    """
    Detailed inspection of Q and K vectors for each word.
    Shows how each dimension contributes to word relationships.
    """
    print(f"\n=== Inspecting Q/K Vectors for: '{text}' ===")
    
    # Convert text to tokens
    tokens = torch.tensor([dataset.vocab.get(word, 0) for word in text.split()])
    words = text.split()
    
    # Get device
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    
    # Enable gradient tracking for intermediate values
    tokens.requires_grad_(True)
    
    # Forward pass with hooks to capture Q and K vectors
    q_vectors = []
    k_vectors = []
    
    def hook_fn(module, input, output):
        # Capture Q and K vectors from MultiHeadAttention
        if isinstance(module, model.blocks[0].attn):
            # Get Q, K vectors before head split
            q = output[0]  # Shape: [batch, seq_len, embd]
            k = output[1]
            q_vectors.append(q.detach())
            k_vectors.append(k.detach())
    
    # Register hook on first attention layer
    hook = model.blocks[0].attn.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        logits, _ = model(tokens.unsqueeze(0))
    
    # Remove hook
    hook.remove()
    
    # Analyze vectors
    q = q_vectors[0][0]  # Shape: [seq_len, embd]
    k = k_vectors[0][0]  # Shape: [seq_len, embd]
    
    print("\nQuery Vector Analysis:")
    print("Each row shows how a word encodes what it's looking for")
    for i, word in enumerate(words):
        print(f"\n{word}:")
        # Show first few dimensions of query vector
        print("First 8 dimensions of query vector:")
        print(q[i, :8].numpy())
    
    print("\nKey Vector Analysis:")
    print("Each row shows how a word advertises itself to others")
    for i, word in enumerate(words):
        print(f"\n{word}:")
        # Show first few dimensions of key vector
        print("First 8 dimensions of key vector:")
        print(k[i, :8].numpy())
    
    print("\nAttention Score Breakdown:")
    # Compute attention scores between specific word pairs
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            # Compute dot product between query of word1 and key of word2
            score = torch.dot(q[i], k[j]) / math.sqrt(q.size(-1))
            print(f"{word1} attending to {word2}: {score.item():.3f}")
            if i == j:  # Same word
                print("  (self-attention score)")
            elif abs(i - j) == 1:  # Adjacent words
                print("  (adjacent word score)")

def analyze_word_relationships():
    """Demonstrate how the model processes and generates text with attention."""
    # Create dataset first to get vocabulary
    block_size = 12
    dataset = SimpleDataset(block_size)
    
    # Create a model focused on interpretability
    config = GPTConfig(
        vocab_size=max(dataset.vocab.values()) + 1,
        block_size=block_size,
        n_layer=2,         # Fewer layers for clearer analysis
        n_head=4,          # Fewer heads to make patterns more distinct
        n_embd=64,         # Smaller embeddings to make inspection manageable
        dropout=0.1,
        debug=True
    )
    
    model = GPT(config)
    print("\n=== Model Architecture ===")
    print("Model configured for interpretable attention analysis")
    print(model)
    
    # Train for longer
    print("\n=== Training on Diverse Sentence Patterns ===")
    train(model, dataset, epochs=10, batch_size=8)
    
    print("\n=== Vector Inspection Points ===")
    print("Analyzing Q/K vectors at key breakpoints:")
    
    # Breakpoint 1: Simple subject-verb
    print("\nBreakpoint 1: Simple subject-verb relationship")
    inspect_vectors(model, "the cat runs", dataset)
    
    # Breakpoint 2: Adjective-noun relationship
    print("\nBreakpoint 2: Adjective-noun relationship")
    inspect_vectors(model, "the quick brown fox", dataset)
    
    # Breakpoint 3: Conjunction relationship
    print("\nBreakpoint 3: Conjunction relationship")
    inspect_vectors(model, "cats and dogs", dataset)
    
    print("\n=== Generating Text with Different Contexts ===")
    
    # Generate with various prompts
    prompts = [
        "the quick",
        "a happy",
        "the cat",
        "dogs and",
        "the small brown"
    ]
    
    for prompt in prompts:
        start_tokens = words_to_tokens(prompt, dataset)
        print(f"\nStarting with '{prompt}':")
        
        print("High temperature (creative):")
        for _ in range(3):
            generated = generate(model, start_tokens, dataset, max_tokens=6, temperature=1.0)
            print(f"Generated: {' '.join(generated)}")
        
        print("\nMedium temperature (balanced):")
        for _ in range(3):
            generated = generate(model, start_tokens, dataset, max_tokens=6, temperature=0.7)
            print(f"Generated: {' '.join(generated)}")
        
        print("\nLow temperature (focused):")
        for _ in range(3):
            generated = generate(model, start_tokens, dataset, max_tokens=6, temperature=0.3)
            print(f"Generated: {' '.join(generated)}")
    
    print("\n=== Testing Long-Range Dependencies ===")
    complex_patterns = [
        "the quick cat and small dog are friends",
        "a happy bird and brown fox run",
        "the small cat is quick and happy"
    ]
    
    print("\nAnalyzing complex sentences with multiple relationships:")
    for text in complex_patterns:
        print(f"\nAnalyzing: '{text}'")
        print("Watch for attention patterns between related words:")
        logits, _, qk_vectors = model(words_to_tokens(text, dataset).unsqueeze(0))
        
        # Analyze Q/K vectors for this sentence
        print("\nAnalyzing Q/K vectors for each word:")
        for layer_name, (q, k) in qk_vectors:
            print(f"\n{layer_name}:")
            words = text.split()
            
            # Show how each word attends to others
            for i, word1 in enumerate(words):
                print(f"\n{word1} attention analysis:")
                q_vec = q[0, i]  # Get query vector for this word
                
                # Compute attention scores with all words
                for j, word2 in enumerate(words):
                    k_vec = k[0, j]  # Get key vector for target word
                    score = torch.dot(q_vec, k_vec) / np.sqrt(q_vec.size(0))
                    print(f"  Attending to {word2}: {score.item():.3f}")

if __name__ == "__main__":
    print("=== Deep Dive into Attention Mechanics ===")
    print("This demo shows how words encode their relationships through Q/K vectors:")
    print("1. Inspect raw vector values")
    print("2. See how dimensions encode different features")
    print("3. Understand attention score computation")
    print("4. Compare self-attention vs cross-attention scores")
    analyze_word_relationships()
