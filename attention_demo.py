import torch
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from model import GPT, GPTConfig
from train import SimpleDataset, train
import os

def ensure_viz_dir():
    """Ensure the visualization directory exists."""
    os.makedirs('attention_viz', exist_ok=True)

def visualize_attention_weights(weights, tokens=None, title="Attention Weights", save_path=None):
    """
    Visualize attention weights as a heatmap with token labels.
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Convert to numpy and get dimensions
    weights_np = weights.detach().cpu().numpy()
    n = weights_np.shape[0]
    
    # Create default token labels if none provided
    if tokens is None:
        tokens = [f"Token {i+1}" for i in range(n)]
    
    # Create heatmap with labels
    sns.heatmap(
        weights_np,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu',
        square=True,
        xticklabels=tokens,
        yticklabels=tokens,
        vmin=0,
        vmax=1,
        center=0.5
    )
    
    plt.xlabel('Attended to (Key)')
    plt.ylabel('Attending from (Query)')
    plt.title(f"{title}\nEach row shows how much a token attends to other tokens")
    plt.tight_layout()
    
    if save_path:
        ensure_viz_dir()  # Ensure directory exists before saving
        save_path = os.path.join('attention_viz', save_path)
        plt.savefig(save_path)
        print(f"Saved attention visualization to {save_path}")
    
    plt.close(fig)

def get_attention_patterns(model, text, dataset):
    """Get attention patterns for a given text."""
    # Convert text to tokens
    tokens = torch.tensor([dataset.vocab.get(word, 0) for word in text.split()])
    words = text.split()
    
    # Forward pass with attention capture
    model.eval()
    with torch.no_grad():
        logits, _, qk_vectors = model(tokens.unsqueeze(0))
    
    return words, qk_vectors

def analyze_attention_evolution():
    """Show how attention patterns evolve during training."""
    # Create dataset and model
    block_size = 12
    dataset = SimpleDataset(block_size)
    
    config = GPTConfig(
        vocab_size=max(dataset.vocab.values()) + 1,
        block_size=block_size,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.1,
        debug=True
    )
    
    # Test sentences that demonstrate different relationships
    test_sentences = [
        "the cat chases the bird",  # Subject-verb-object
        "the quick brown fox",      # Adjective chain
        "cats and dogs are friends" # Conjunction
    ]
    
    print("=== Analyzing Attention Pattern Evolution ===")
    
    # Create and analyze untrained model
    print("\nStep 1: Untrained Model Patterns")
    model = GPT(config)
    
    print("\nUntrained attention patterns:")
    for text in test_sentences:
        print(f"\nAnalyzing: '{text}'")
        words, qk_vectors = get_attention_patterns(model, text, dataset)
        
        # Show attention patterns for each layer
        for layer_idx, (layer_name, (q, k)) in enumerate(qk_vectors):
            print(f"\n{layer_name}:")
            
            # Compute attention scores
            att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            att = F.softmax(att, dim=-1)
            
            # Save visualization
            save_path = f"attention_{text.replace(' ', '_')}_untrained_layer_{layer_idx+1}.png"
            visualize_attention_weights(
                att[0],
                words,
                f"Untrained Model - Layer {layer_idx+1}",
                save_path=save_path
            )
    
    # Train the model in stages and analyze patterns
    print("\nStep 2: Training Evolution")
    training_stages = [1, 2, 3, 5, 10]  # More frequent checks in early training
    
    for stage in training_stages:
        print(f"\n=== After {stage} epochs ===")
        epochs_to_train = int(stage - model.current_epoch.item()) if stage > model.current_epoch.item() else 0
        train(model, dataset, epochs=epochs_to_train, batch_size=8)
        model.current_epoch = torch.tensor(stage)
        
        print(f"\nAnalyzing attention patterns after {stage} epochs:")
        for text in test_sentences:
            print(f"\nAnalyzing: '{text}'")
            words, qk_vectors = get_attention_patterns(model, text, dataset)
            
            # Show attention patterns for each layer
            for layer_idx, (layer_name, (q, k)) in enumerate(qk_vectors):
                print(f"\n{layer_name}:")
                
                # Compute attention scores
                att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
                att = F.softmax(att, dim=-1)
                
                # Show what each word is attending to
                for i, word in enumerate(words):
                    print(f"\n{word} attends to:")
                    for j, other_word in enumerate(words):
                        score = att[0, i, j].item()
                        if score > 0.1:  # Only show significant attention
                            print(f"  {other_word}: {score:.3f}")
                
                # Save visualization
                stage_name = f"epoch_{stage}"
                save_path = f"attention_{text.replace(' ', '_')}_{stage_name}_layer_{layer_idx+1}.png"
                
                print(f"\nAnalyzing attention patterns:")
                print("Look for:")
                if "the" in words:
                    print("- Article 'the' attending to following nouns")
                if any(w in words for w in ["quick", "brown", "small"]):
                    print("- Adjectives attending to their nouns")
                if "and" in words:
                    print("- 'and' attending equally to words it connects")
                if "chases" in words:
                    print("- Verb attending to subject and object")
                
                visualize_attention_weights(
                    att[0],
                    words,
                    f"After {stage} epochs - Layer {layer_idx+1}",
                    save_path=save_path
                )

def demonstrate_specific_patterns():
    """Show specific attention patterns as they develop during training."""
    # Create dataset and model
    block_size = 12
    dataset = SimpleDataset(block_size)
    
    config = GPTConfig(
        vocab_size=max(dataset.vocab.values()) + 1,
        block_size=block_size,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.1,
        debug=True
    )
    
    model = GPT(config)
    model.current_epoch = torch.tensor(0)  # Track training progress
    
    # Training stages
    stages = [0, 3, 7, 10]  # Check at start, early, middle, and end of training
    
    # Define patterns to analyze
    patterns = [
        ("Article -> Noun", "the cat", 
         "Shows how articles learn to attend to their nouns"),
        
        ("Adjective -> Noun", "quick brown fox", 
         "Shows how adjectives learn to modify nouns"),
        
        ("Subject -> Verb", "cat chases bird", 
         "Shows how subject-verb relationships develop"),
        
        ("Conjunction", "cat and dog", 
         "Shows how conjunctions learn to link words"),
    ]
    
    print("\nAnalyzing how attention patterns develop during training:")
    for stage in stages:
        print(f"\n=== Training Stage: {stage} epochs ===")
        
        # Train to this stage
        if stage > model.current_epoch:
            epochs_to_train = int(stage - model.current_epoch.item())
            print(f"\nTraining for {epochs_to_train} more epochs...")
            train(model, dataset, epochs=epochs_to_train, batch_size=8)
            model.current_epoch = torch.tensor(stage)
    
    print("\nAnalyzing learned attention patterns:")
    for pattern_name, text, description in patterns:
        print(f"\n=== {pattern_name} ===")
        print(f"Text: '{text}'")
        print(f"Expected: {description}")
        
        words, qk_vectors = get_attention_patterns(model, text, dataset)
        
        # Show attention patterns for each layer
        for layer_idx, (layer_name, (q, k)) in enumerate(qk_vectors):
            # Compute attention scores
            att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            att = F.softmax(att, dim=-1)
            
            # Save visualization
            save_path = f"attention_{pattern_name.lower().replace(' -> ', '_')}_{stage}_layer_{layer_idx+1}.png"
            visualize_attention_weights(
                att[0],
                words,
                f"{pattern_name} - Layer {layer_idx+1}",
                save_path=save_path
            )
            
            # Print attention analysis
            print(f"\n{layer_name} Analysis:")
            for i, word in enumerate(words):
                print(f"\n{word} attends to:")
                for j, other_word in enumerate(words):
                    score = att[0, i, j].item()
                    if score > 0.1:
                        print(f"  {other_word}: {score:.3f}")

if __name__ == "__main__":
    # Clean up any existing visualizations
    if os.path.exists('attention_viz'):
        print("Cleaning up old visualizations...")
        import shutil
        shutil.rmtree('attention_viz')
    ensure_viz_dir()  # Create fresh visualization directory
    
    print("=== Attention Pattern Analysis ===")
    print("This demo shows how attention patterns evolve with training")
    print("and how different linguistic relationships are captured")
    print("Visualizations will be saved in the 'attention_viz' directory")
    
    try:
        print("\nPart 1: Evolution of Attention Patterns")
        analyze_attention_evolution()
        
        print("\nPart 2: Specific Learned Patterns")
        demonstrate_specific_patterns()
        
        print("\nAnalysis complete! Visualizations saved in 'attention_viz' directory")
        print("To clean up visualizations, delete the 'attention_viz' directory")
    except KeyboardInterrupt:
        print("\nCleaning up...")
        if os.path.exists('attention_viz'):
            shutil.rmtree('attention_viz')
        print("Cleanup complete")
