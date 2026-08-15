### TRANSFORMER ARCHITECTURE ###

import torch
import torch.nn as nn

token_attributes = ["family", "bar_position", "channel", "program", "pitch", "velocity", "duration", "tempo", "time_signature", "controller_type", "controller_value"]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
      super(TransformerBlock, self).__init__()

      self.norm1 = nn.LayerNorm(d_model)

      # internally, the tensor is viewed approximately as (N, H, L, D_head) where H is the # of heads and D_head is the amount of embedding dimensions a head attends to
      self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

      self.attn_dropout = nn.Dropout(dropout)

      self.norm2 = nn.LayerNorm(d_model)

      self.feed_forward = nn.Sequential(
          nn.Linear(d_model, 4 * d_model),
          nn.GELU(),
          nn.Dropout(dropout),
          nn.Linear(4 * d_model, d_model),
          nn.Dropout(dropout)
      )

    def forward(self, x, padding_mask, mode):
      T = x.size(1)

      # pre-norm architecture
      residual = x

      x = self.norm1(x)

      if mode == "decode":

        """
        Attention Matrix and Mask
        Q/K   Tok1 Tok2 Tok3 Tok4

        Tok1   ✓    x    x    x
        Tok2   ✓    ✓    x    x
        Tok3   ✓    ✓    ✓    x
        Tok4   ✓    ✓    ✓    ✓
        """

        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_output, _ = self.attention(x, x, x, attn_mask=causal_mask, key_padding_mask=padding_mask, need_weights=False)

      # shape of attn_output: (N, L, E)

      x = residual + self.attn_dropout(attn_output)

      residual = x

      x = self.norm2(x)

      ff_output = self.feed_forward(x)

      x = residual + ff_output

      return x

def make_transformer_decoder(vocab_sizes, attribute_embedding_dim=32, block_size=1024, d_model=384, num_heads=6, dropout=0.1, num_layers=6):
  "Create a transformer decoder"
  class TransformerDecoder(nn.Module):
    def __init__ (self, vocab_sizes, attribute_embedding_dim=32, block_size=1024, d_model=384, num_heads=6, dropout=0.1, num_layers=6):
      super().__init__()

      # define blocks/layers
      self.attribute_embeddings = nn.ModuleDict({field: nn.Embedding(num_embeddings=vocab_sizes[field], embedding_dim=attribute_embedding_dim) for field in token_attributes})

      concatenated_dimension = attribute_embedding_dim * len(token_attributes)
      self.input_projection = nn.Linear(concatenated_dimension, d_model)

      self.num_layers = num_layers
      self.d_model = d_model

      self.position_embedding = nn.Embedding(block_size, d_model)
      self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)])

      self.norm = nn.LayerNorm(d_model)

      # One output classifier/head for every field
      self.output_heads = nn.ModuleDict({field: nn.Linear(d_model, vocab_sizes[field]) for field in token_attributes})

    def forward(self, x):
      # define how activations move
      # x is of shape (B, T, F)
      B, T, number_of_fields = x.shape
      if number_of_fields != len(token_attributes): raise ValueError(f"Expected {len(token_attributes)} fields, received {number_of_fields}")
      if T > self.position_embedding.num_embeddings: raise ValueError("Input sequence is longer than block_size")

      # Construct padding mask
      padding_mask = x[:, :, 0] == 0

      embedded_fields = []
      for i in range(number_of_fields):
        embedded_fields.append(self.attribute_embeddings[token_attributes[i]](x[:, :, i]))

      # Concatenate along the final dimension: 11 tensors of shape (B, T, attribute_embedding_dim) -> (B, T, attribute_embedding_dim*11)
      x = torch.cat(embedded_fields, dim=-1)

      # Project into d_model
      x = self.input_projection(x)

      # Positional embedding
      positions = torch.arange(T, device=x.device)
      x = x + self.position_embedding(positions)

      # Decode
      for block in self.transformer_blocks:
        x = block(x, padding_mask, "decode")

      x = self.norm(x)

      logits = {field: self.output_heads[field](x) for field in token_attributes}

      return logits

  model = TransformerDecoder(vocab_sizes, attribute_embedding_dim, block_size, d_model, num_heads, dropout, num_layers)

  return model