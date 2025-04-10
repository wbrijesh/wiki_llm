"""
Transformer Language Model implementation based on "Attention is All You Need" paper
https://arxiv.org/abs/1706.03762
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism.

    Computes attention as described in the original transformer paper:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for scaled dot-product attention.

        Args:
            query: Tensor of shape (batch_size, num_heads, seq_len, d_k)
            key: Tensor of shape (batch_size, num_heads, seq_len, d_k)
            value: Tensor of shape (batch_size, num_heads, seq_len, d_v)
            mask: Optional mask tensor (1 = keep, 0 = mask out)

        Returns:
            output: Weighted sum based on attention scores
            attention_weights: The attention distribution
        """
        # Get dimensions
        d_k = query.size(-1)

        # Compute attention scores
        # (batch_size, num_heads, seq_len, d_k) x (batch_size, num_heads, d_k, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (if provided)
        if mask is not None:
            # Fill masked positions with very small values (effectively zero after softmax)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute weighted sum of values
        # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, d_v)
        # -> (batch_size, num_heads, seq_len, d_v)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention as described in "Attention is All You Need"

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_o
    where head_i = Attention(Q * W_q_i, K * W_k_i, V * W_v_i)
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension/embedding size
            num_heads: Number of attention heads
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model  # Embedding dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head

        # Linear projections for Q, K, V for all heads (in a batch)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention()

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k)
        and reshape to (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_length, _ = x.size()

        # Reshape to (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)

        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transpose and reshape from (batch_size, num_heads, seq_len, d_k)
        to (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_length, _ = x.size()

        # Transpose to (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2).contiguous()

        # Reshape to (batch_size, seq_len, d_model)
        return x.view(batch_size, seq_length, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.

        Args:
            query: Input tensor of shape (batch_size, seq_len, d_model)
            key: Input tensor of shape (batch_size, seq_len, d_model)
            value: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights
        """
        # batch_size = query.size(0)

        # Project inputs
        query = self.W_q(query)  # (batch_size, seq_len, d_model)
        key = self.W_k(key)      # (batch_size, seq_len, d_model)
        value = self.W_v(value)  # (batch_size, seq_len, d_model)

        # Split heads
        query = self.split_heads(query)  # (batch_size, num_heads, seq_len, d_k)
        key = self.split_heads(key)      # (batch_size, num_heads, seq_len, d_k)
        value = self.split_heads(value)  # (batch_size, num_heads, seq_len, d_v)

        # Apply scaled dot-product attention
        attn_output, attention_weights = self.attention(query, key, value, mask)

        # Combine heads
        attn_output = self.combine_heads(attn_output)  # (batch_size, seq_len, d_model)

        # Final linear projection
        output = self.W_o(attn_output)  # (batch_size, seq_len, d_model)

        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network from "Attention is All You Need"

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Hidden dimension of feed-forward network
        """
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)   # First linear layer
        self.fc2 = nn.Linear(d_ff, d_model)   # Second linear layer
        self.relu = nn.ReLU()                 # ReLU activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for position-wise feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """Positional Encoding as described in "Attention is All You Need"

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)

        # Create position indices: (max_seq_length, 1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # Create division term: (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_seq_length, d_model)
        pe = pe.unsqueeze(0)

        # Register buffer (not a parameter but part of the module)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor with positional encoding added
        """
        # Cast buffer to tensor explicitly for type checkers
        pe = torch.as_tensor(self.pe)

        # Add positional encoding to input embeddings
        x = x + pe[:, :x.size(1), :]

        return self.dropout(x)


class DecoderLayer(nn.Module):
    """Single decoder layer from "Attention is All You Need"

    Contains:
    1. Masked Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network
    3. Layer Normalization and Residual Connections
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize decoder layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()

        # Self-attention layer
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # Feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for decoder layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerLM(nn.Module):
    """Transformer-based Language Model, similar to GPT architecture"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize transformer language model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension/embedding size
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            d_ff: Hidden dimension of feed-forward network
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final linear layer for token prediction
        self.final_layer = nn.Linear(d_model, vocab_size)

        # Store model dimensions
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters with Xavier uniform distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """
        Generate a square causal attention mask.

        The mask ensures that the predictions for position i
        can only depend on known outputs at positions less than i.

        Args:
            size: Size of the square mask

        Returns:
            Causal mask tensor
        """
        # Create a lower triangular matrix (where upper triangle is 0)
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()

        # Convert to appropriate format for attention (upper triangle = -inf)
        mask = torch.zeros(size, size).masked_fill(mask, float('-inf'))

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer language model.

        Args:
            x: Input token IDs of shape (batch_size, seq_len)

        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_length = x.size()

        # Create causal mask to prevent attending to future positions
        device = x.device
        mask = self.generate_square_subsequent_mask(seq_length).to(device)

        # Convert token indices to embeddings
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.token_embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mask)

        # Final linear layer to predict next token
        output = self.final_layer(x)

        return output

    def get_num_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        self.eval()
        tokens = start_tokens.unsqueeze(0) if start_tokens.dim() == 1 else start_tokens

        for _ in range(max_tokens):
            # Get logits for next token
            logits = self(tokens)[:, -1, :] / temperature

            # Apply nucleus sampling (top-p)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = float('-inf')

            # Sample from adjusted distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append new token
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens.squeeze(0)
