import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load configuration from config.json
import os
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config.json'))
with open(config_path, 'r') as f:
    config_data = json.load(f)

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for transformer models.

    This class implements the multi-head attention mechanism, which allows the model
    to focus on different parts of the input sequence simultaneously. It includes
    key, query, and value projections, as well as dropout for regularization.

    Args:
        config (GPTConfig): Configuration object containing model hyperparameters.
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.debug = config.debug

        # key, query, value projections
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).
            mask (torch.Tensor, optional): Mask tensor to prevent attention to certain positions.

        Returns:
            torch.Tensor: Output tensor after applying attention.
            tuple: Raw query and key vectors if debug mode is enabled.
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate raw Q, K, V projections (before head split)
        q_raw = self.query(x)  # (B, T, embd)
        k_raw = self.key(x)    # (B, T, embd)
        v_raw = self.value(x)  # (B, T, embd)
        
        # Split into heads
        k = k_raw.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q_raw.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v_raw.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)

        if self.debug:
            print(f"\nAttention Debug:")
            print(f"Input shape: {x.shape}")
            print(f"Raw projections (before head split):")
            print(f"Q shape: {q_raw.shape}")
            print(f"K shape: {k_raw.shape}")
            print(f"V shape: {v_raw.shape}")
            print(f"\nAfter head split:")
            print(f"Q shape: {q.shape}")
            print(f"K shape: {k.shape}")
            print(f"V shape: {v.shape}")

        # Compute attention scores and apply attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if mask is not None:
            att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        if self.debug:
            print(f"\nAttention scores:")
            print(f"Shape: {att.shape}")
            print(f"Sample (first head):\n{att[0, 0]}")
        
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        
        if self.debug:
            # Return both attention outputs and raw Q,K vectors for inspection
            return y, (q_raw, k_raw)
        return y

class Block(nn.Module):
    """
    Transformer block consisting of multi-head attention and feed-forward layers.

    This class implements a single transformer block, which includes layer normalization,
    multi-head attention, and a feed-forward neural network. It also supports debug mode
    for inspecting intermediate values.

    Args:
        config (GPTConfig): Configuration object containing model hyperparameters.
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
        self.debug = config.debug

    def forward(self, x):
        """
        Forward pass for the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor after processing through the block.
            tuple: Q/K vectors if debug mode is enabled.
        """
        if self.debug:
            print(f"\nBlock Input shape: {x.shape}")
        
        # Layer norm and attention
        x_norm = self.ln1(x)
        if self.debug:
            print(f"After LayerNorm1 shape: {x_norm.shape}")
            print(f"Sample normalized values:\n{x_norm[0, 0, :5]}")
        
        # Handle debug mode return values from attention
        if self.debug:
            attn_output, qk_vectors = self.attn(x_norm)
            # Residual connection
            x = x + attn_output
            print(f"\nAfter attention + residual shape: {x.shape}")
            print(f"Sample values:\n{x[0, 0, :5]}")
        else:
            attn_output = self.attn(x_norm)
            # Residual connection
            x = x + attn_output
        
        # Layer norm and MLP
        x_norm = self.ln2(x)
        mlp_output = self.mlp(x_norm)
        # Residual connection
        x = x + mlp_output
        
        if self.debug:
            print(f"\nFinal block output shape: {x.shape}")
            print(f"Sample output values:\n{x[0, 0, :5]}")
            # Return both the output and the Q/K vectors for inspection
            return x, qk_vectors
        
        return x

class GPT(nn.Module):
    """
    Generative Pre-trained Transformer (GPT) model.

    This class implements the GPT model, which consists of an embedding layer, multiple
    transformer blocks, and a final linear layer for output. It supports debug mode for
    inspecting intermediate values and attention patterns.

    Args:
        config (GPTConfig): Configuration object containing model hyperparameters.
    """
    def __init__(self, config):
        super().__init__()
        self.debug = config.debug
        
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        
        # training progress tracking
        self.register_buffer('current_epoch', torch.tensor(0))
        
        # transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for the model layers.

        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        """
        Forward pass for the GPT model.

        Args:
            idx (torch.Tensor): Input indices tensor of shape (batch_size, sequence_length).
            targets (torch.Tensor, optional): Target indices tensor for calculating loss.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, sequence_length, vocab_size).
            torch.Tensor: Loss value if targets are provided.
            list: Q/K vectors if debug mode is enabled.
        """
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.debug:
            print(f"\nGPT Forward Pass:")
            print(f"Input indices shape: {idx.shape}")

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        
        if self.debug:
            print(f"\nAfter embedding + positional:")
            print(f"Token embeddings shape: {token_embeddings.shape}")
            print(f"Position embeddings shape: {position_embeddings.shape}")
            print(f"Combined embeddings shape: {x.shape}")
            print(f"Sample embedded values:\n{x[0, 0, :5]}")

        # Store Q/K vectors from each layer if in debug mode
        qk_vectors = []
        
        for i, block in enumerate(self.blocks):
            if self.debug:
                x, layer_qk = block(x)
                qk_vectors.append((f"Layer {i}", layer_qk))
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        if self.debug:
            print(f"\nFinal output:")
            print(f"Logits shape: {logits.shape}")
            print(f"Sample logits:\n{logits[0, 0, :5]}")
            
            # Print Q/K vector analysis for each layer
            print("\nQ/K Vector Analysis Across Layers:")
            for layer_name, (q, k) in qk_vectors:
                print(f"\n{layer_name}:")
                print(f"Q shape: {q.shape}, K shape: {k.shape}")
                print(f"Sample Q vector (first token):\n{q[0, 0, :5]}")
                print(f"Sample K vector (first token):\n{k[0, 0, :5]}")

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if self.debug:
            # Return Q/K vectors along with normal outputs
            return logits, loss, qk_vectors
        return logits, loss

class GPTConfig:
    """
    Configuration class for the GPT model.

    This class stores the hyperparameters for the GPT model, including vocabulary size,
    block size, number of layers, number of attention heads, embedding dimension, dropout
    rate, and debug mode.

    Args:
        vocab_size (int): Size of the vocabulary.
        block_size (int): Maximum sequence length.
        n_layer (int): Number of transformer layers.
        n_head (int): Number of attention heads.
        n_embd (int): Embedding dimension.
        dropout (float): Dropout rate for regularization.
        debug (bool): Whether to enable debug mode.
    """
    def __init__(self, vocab_size, block_size, n_layer=6, n_head=8, n_embd=512,
                 dropout=0.1, debug=False):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.debug = debug
