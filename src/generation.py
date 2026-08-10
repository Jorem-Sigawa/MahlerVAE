### GENERATION ###
import os

import torch
from tqdm import tqdm
from .generation_constraints import CompoundREMIGenerationConstraints, generate_constrained_tokens

token_attributes = ["family", "bar_position", "channel", "program", "pitch", "velocity", "duration", "tempo", "time_signature", "controller_type", "controller_value"]

def continue_text(
  model,
  out_path,
  prompt_path,
  bar_start,
  generation_length,
  batch_size,
  params,
  device,
  field2idx,
  idx2field,
  max_tokens_without_metric=24,
  temperature=0.9,
  top_k=8,
  min_velocity=6,
  min_duration=3,
  positions_per_beat=12,
  sample_latent=False
):

  # set model to eval
  model.eval()

  with open(prompt_path, "r") as f:
    prompt_tokens = [line.strip().split() for line in f if line.strip()]

  prompt_text = []
  current_bar = 0

  for token in prompt_tokens:
    if token[1] == "BAR":
      current_bar += 1
      if current_bar >= bar_start:
        break

    prompt_text.append(token)

  if len(prompt_text) == 0:
    raise ValueError("The prompt does not contain any tokens before bar_start.")

  current_numerator = 4
  current_denominator = 4

# Find initial time signature of prompt
  for token in prompt_text:
    if token[0] == "Metric" and token[1] == "BAR" and token[8] not in ("IGNORE", "CONTINUE"):
      try:
        current_numerator, current_denominator = map(int, token[8].split("/"))
      except (ValueError, IndexError):
        pass

  positions_per_bar = max(1, round(current_numerator * 4 * positions_per_beat / current_denominator))

  def vectorize_prompt(prompt_text):
    tokenized_prompt = [ [field2idx[field].get(token[i], field2idx[field]["UNK"]) for i, field in enumerate(token_attributes)] for token in prompt_text ]
    return torch.tensor(tokenized_prompt, dtype=torch.long, device=device).unsqueeze(0)
  
  prompt = vectorize_prompt(prompt_text)
  encoder_prompt = prompt[:, -params["seq_length"]:]

  with torch.inference_mode():
    z_mean, z_logvar = model.encode(encoder_prompt)
    z_mean = z_mean.repeat(batch_size, 1)
    z_logvar = z_logvar.repeat(batch_size, 1)
    z = model.reparameterize(z_mean, z_logvar) if sample_latent else z_mean

    # Codex-assisted stable generation
    constraints = CompoundREMIGenerationConstraints(
      field2idx=field2idx,
      token_attributes=token_attributes,
      batch_size=batch_size,
      positions_per_bar=positions_per_bar,
      device=device,
      max_tokens_without_metric=max_tokens_without_metric,
      temperature=temperature,
      top_k=top_k,
      min_velocity=min_velocity,
      min_duration=min_duration
    )
    generated_tokens = generate_constrained_tokens(
      model=model,
      prompt=prompt,
      latent_provider=lambda bars_generated: z,
      generation_length=generation_length,
      sequence_length=params["seq_length"],
      constraints=constraints,
      idx2field=idx2field,
      progress=tqdm
    )

  prompt_lines = [" ".join(token) for token in prompt_text]
  for sample_idx in range(batch_size):
    with open(f"{out_path}{sample_idx}.txt", "w") as f:
      f.write("\n".join(prompt_lines + generated_tokens[sample_idx]))

def continue_text_with_interpolation(
  model,
  out_path,
  encoded_prompt1,
  encoded_prompt2,
  generation_length,
  batch_size,
  params,
  device,
  field2idx,
  idx2field,
  interpolation_start_bar=4,
  interpolation_length_bars=24,
  max_tokens_without_metric=24,
  temperature=0.9,
  top_k=8,
  min_velocity=6,
  min_duration=3,
  positions_per_beat=12,
  sample_latent=False
):
  # set model to eval
  model.eval()

  with open(encoded_prompt1, "r") as f:
    prompt1_text = [line.strip().split() for line in f if line.strip()]

  with open(encoded_prompt2, "r") as f:
    prompt2_text = [line.strip().split() for line in f if line.strip()]

  if len(prompt1_text) == 0:
    raise ValueError("encoded_prompt1 does not contain any tokens.")
  if len(prompt2_text) == 0:
    raise ValueError("encoded_prompt2 does not contain any tokens.")

  current_numerator = 4
  current_denominator = 4

  # Find initial time signature of prompt
  for token in prompt1_text:
    if token[0] == "Metric" and token[1] == "BAR" and token[8] not in ("IGNORE", "CONTINUE"):
      try:
        current_numerator, current_denominator = map(int, token[8].split("/"))
      except (ValueError, IndexError):
        pass

  positions_per_bar = max(1, round(current_numerator * 4 * positions_per_beat / current_denominator))

  def vectorize_prompt(prompt_text):
    tokenized_prompt = [ [field2idx[field].get(token[i], field2idx[field]["UNK"]) for i, field in enumerate(token_attributes)] for token in prompt_text ]
    return torch.tensor(tokenized_prompt, dtype=torch.long, device=device).unsqueeze(0)

  prompt1 = vectorize_prompt(prompt1_text)
  prompt2 = vectorize_prompt(prompt2_text)

  # Get last seq_length tokens only
  encoder_prompt1 = prompt1[:, -params["seq_length"]:]
  encoder_prompt2 = prompt2[:, -params["seq_length"]:]

  with torch.inference_mode():
    z_mean1, z_logvar1 = model.encode(encoder_prompt1)
    z_mean2, z_logvar2 = model.encode(encoder_prompt2)

    # shape: (1, latent_dim) -> (batch_size, latent_dim)
    z_mean1 = z_mean1.repeat(batch_size, 1)
    z_logvar1 = z_logvar1.repeat(batch_size, 1)
    z_mean2 = z_mean2.repeat(batch_size, 1)
    z_logvar2 = z_logvar2.repeat(batch_size, 1)

    if sample_latent:
      z1 = model.reparameterize(z_mean1, z_logvar1)
      z2 = model.reparameterize(z_mean2, z_logvar2)
    else:
      z1 = z_mean1
      z2 = z_mean2

    # bars_generated.shape: (batch_size)
    def interpolated_latent(bars_generated):
      alpha = ((bars_generated.float() - interpolation_start_bar) / interpolation_length_bars).clamp(0.0, 1.0).unsqueeze(1)
      # Why .unsqueeze(1)?
      # PyTorch broadcasting works by matching dimensions from the right. Without .unsqueeze(1),
      # alpha:             [batch_size]
      # z2-z1: [batch_size, latent_dim]
      return z1 + alpha * (z2 - z1)

    # Codex-assisted stable generation
    constraints = CompoundREMIGenerationConstraints(
      field2idx=field2idx,
      token_attributes=token_attributes,
      batch_size=batch_size,
      positions_per_bar=positions_per_bar,
      device=device,
      max_tokens_without_metric=max_tokens_without_metric,
      temperature=temperature,
      top_k=top_k,
      min_velocity=min_velocity,
      min_duration=min_duration
    )
    generated_tokens = generate_constrained_tokens(
      model=model,
      prompt=prompt1,
      latent_provider=interpolated_latent,
      generation_length=generation_length,
      sequence_length=params["seq_length"],
      constraints=constraints,
      idx2field=idx2field,
      progress=tqdm
    )

  prompt_lines = [" ".join(token) for token in prompt1_text]
  for sample_idx in range(batch_size):
    with open(f"{out_path}{sample_idx}.txt", "w") as f:
      f.write("\n".join(prompt_lines + generated_tokens[sample_idx]))