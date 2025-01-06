import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import GPT, GPTConfig, MultiHeadAttention

class SimpleDataset(Dataset):
    def __init__(self, block_size):
        # Create a vocabulary of simple words
        self.vocab = {
            # Articles
            'the': 1, 'a': 2,
            # Nouns (animals)
            'cat': 3, 'dog': 4, 'bird': 5, 'fox': 6,
            # Verbs
            'is': 7, 'are': 8, 'chases': 9, 'jumps': 10, 'runs': 11,
            # Adjectives
            'quick': 12, 'brown': 13, 'happy': 14, 'small': 15,
            # Conjunctions
            'and': 16,
            # Other
            'friends': 17
        }
        
        # Create a diverse set of training sentences
        sentences = [
            # Simple SVO patterns
            "the cat chases the bird",
            "the dog chases the cat",
            "a bird runs",
            "the fox jumps",
            "a cat runs",
            
            # Multiple adjectives
            "the quick brown fox jumps",
            "a happy small dog runs",
            "the small brown bird jumps",
            "a quick happy cat runs",
            "the small quick fox jumps",
            
            # Compound subjects with conjunctions
            "cats and dogs are friends",
            "the cat and bird are friends",
            "the dog and fox are quick",
            "birds and cats are small",
            
            # State descriptions
            "the cat is happy",
            "the dogs are quick",
            "the bird is small",
            "the fox is brown",
            
            # Complex patterns
            "the quick cat and small dog are friends",
            "a happy bird and brown fox run",
            "the small cat is quick and happy",
            "a brown dog and quick fox jump",
            
            # Repeated for better learning
            "the cat chases the bird",
            "the quick brown fox jumps",
            "cats and dogs are friends",
            "the cat is happy",
            "the quick cat and small dog are friends"
        ]
        
        # Convert sentences to token sequences
        self.sequences = []
        for sentence in sentences:
            tokens = [self.vocab.get(word, 0) for word in sentence.split()]
            if len(tokens) < block_size:
                tokens.extend([0] * (block_size - len(tokens)))
            self.sequences.extend([tokens[i:i+block_size] for i in range(len(tokens)-block_size+1)])
        
        self.data = torch.tensor(self.sequences)
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def train(model, train_dataset, epochs=5, batch_size=4, learning_rate=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("\nTraining the model to understand word relationships...")
    print("This will help it learn patterns like:")
    print("- Articles followed by nouns")
    print("- Adjectives describing nouns")
    print("- Subject-verb-object structures")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass (handle debug mode return values)
            if model.debug:
                logits, loss, _ = model(x, y)
            else:
                logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:  # More frequent updates
                print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch+1}/{epochs} Summary:')
        print(f'Average Loss: {avg_loss:.4f}')
        print("The model is learning to predict words based on context")

def generate(model, start_sequence, dataset=None, max_tokens=10, temperature=1.0):
    """
    Generate a sequence of tokens given a start sequence.
    
    Args:
        model: The GPT model to use for generation
        start_sequence: Initial sequence of tokens
        dataset: Optional dataset containing vocabulary for word conversion
        max_tokens: Maximum number of tokens to generate
        temperature: Controls randomness in generation (higher = more random)
    
    Returns:
        List of generated tokens or words if dataset with vocabulary is provided
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x = start_sequence.to(device)
    generated = []
    
    # Get vocabulary if available
    vocab = dataset.vocab if dataset and hasattr(dataset, 'vocab') else None
    rev_vocab = {v: k for k, v in vocab.items()} if vocab else None
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass
            logits, _ = model(x.unsqueeze(0))
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            generated.append(token_id)
            
            # Update input sequence
            x = torch.cat([x, next_token.squeeze(0)])
    
    # Convert tokens to words if vocabulary is available
    if rev_vocab:
        return [rev_vocab.get(token, f'<{token}>') for token in generated]
    return generated

def inspect_attention(model, input_sequence):
    """
    Run a forward pass with debug=True to inspect attention patterns and Q/K vectors
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.debug = True  # Enable debug mode
    
    with torch.no_grad():
        x = input_sequence.to(device)
        logits, _, qk_vectors = model(x.unsqueeze(0))
    
    model.debug = False  # Disable debug mode
    return logits, qk_vectors
