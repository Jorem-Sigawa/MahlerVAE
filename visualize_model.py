import os
import torch

from torchview import draw_graph
from src.model import TransformerVAE, token_attributes

vocab_sizes = {field: 64 for field in token_attributes}

#model = TransformerVAE(
#    vocab_sizes=vocab_sizes,
#    latent_dim=params["latent_dim"],
#    attribute_embedding_dim=params["attribute_embedding_dim"],
#    block_size=params["seq_length"],
#    d_model=params["d_model"],
#    num_heads=params["num_heads"],
#    dropout=params["dropout"],
#    num_layers=params["num_layers"]
#)

model = TransformerVAE(
    vocab_sizes=vocab_sizes,
    latent_dim=64,
    attribute_embedding_dim=32,
    block_size=1024,
    d_model=256,
    num_heads=8,
    dropout=0.1,
    num_layers=4
)

# Dummy input

x = torch.randint(1, 64, (1, 32, len(token_attributes)))

model_graph = draw_graph(
    model,
    input_data=x,
    graph_name="MahlerVAE",
    depth=3,
    expand_nested=True,
    roll=True,
    show_shapes=True,
    graph_dir="LR"
)

model_graph.visual_graph.format = "png"
model_graph.visual_graph.render("assets/architecture", cleanup=True)

print("Model architecture visualization saved to assets/architecture.png")