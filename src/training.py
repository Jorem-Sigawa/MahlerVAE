### TRAINING AND INSTANTIATING MODEL (Defining get_batches, loss, annealing, and training loop) ###

import os
import comet_ml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from IPython import display as ipythondisplay

token_attributes = ["family", "bar_position", "channel", "program", "pitch", "velocity", "duration", "tempo", "time_signature", "controller_type", "controller_value"]

def get_batch(vectorized_songs, seq_length, batch_size):
  # Pick {batch_size} number of tracks
  rng = np.random.default_rng()
  track_idxs = rng.choice(len(vectorized_songs), size=batch_size, replace=True)

  # Sample {seq_length} tokens from each of the tracks
  input_batch = []
  output_batch = []
  for idx in track_idxs:
    n = np.size(vectorized_songs[idx], 0) - 1
    sample_idx = np.random.randint(0, n - seq_length + 1)
    input_batch.append(vectorized_songs[idx][sample_idx:sample_idx + seq_length])
    output_batch.append(vectorized_songs[idx][sample_idx + 1: sample_idx + 1 + seq_length])

  # Apply padding
  input_batch = [torch.tensor(item, dtype=torch.long) for item in input_batch]
  output_batch = [torch.tensor(item, dtype=torch.long) for item in output_batch]

  input_batch = pad_sequence(input_batch, batch_first=True, padding_value=0)
  output_batch = pad_sequence(output_batch, batch_first=True, padding_value=0)

  return input_batch, output_batch

# Find reconstruction loss
field_weights = {
    "family": 2.0,
    "bar_position": 1.5,
    "channel": 0.5,
    "program": 0.5,
    "pitch": 4.0,
    "velocity": 0.75,
    "duration": 2.0,
    "tempo": 0.5,
    "time_signature": 0.5,
    "controller_type": 0.25,
    "controller_value": 0.25,
}

cross_entropy = nn.CrossEntropyLoss(reduction="none")
def reconstruction_loss(labels, logits, field2idx): #labels shape: (B, T, F)
  field_losses = {}
  for i, field in enumerate(token_attributes):
    field_labels = labels[:, :, i] # shape: (B, T)
    field_logits = logits[field] # shape: (B, T, vocab_size[field])

    flattened_labels = field_labels.reshape(-1) # shape: (B*T)
    flattened_logits = field_logits.reshape(-1, field_logits.size(-1)) # shape: (B*T, vocab_size[field])

    field_loss = cross_entropy(flattened_logits, flattened_labels) # per token loss; shape: (B*T)

    # Do not account for "IGNORE" field labels and pad token when computing the loss
    field_loss = field_loss.reshape(field_labels.shape) # shape: (B, T)
    ignore_idx = field2idx[field]["IGNORE"]
    padding_mask = (field_labels != 0) # pad_idx is 0
    valid_mask = (field_labels != ignore_idx) & padding_mask
    masked_field_loss = field_loss[valid_mask]

    if valid_mask.any():
      field_losses[field] = masked_field_loss.mean()

  active_weight_sum = sum(field_weights[field] for field in field_losses)
  total_loss = sum(loss * field_weights[field] for field, loss in field_losses.items()) / active_weight_sum

  return total_loss, field_losses

def kl_loss(mu, logvar, dim_target_kl=0.5):
  mu, logvar = mu.float(), logvar.float() # compute KL in float32 to avoid overflow from exp(logvar)
  kl_per_dim = 0.5*(mu.pow(2) + logvar.exp() - 1 - logvar).mean(dim=0)
  kl_mask = (kl_per_dim > dim_target_kl).float() # KL thresholding
  return (kl_per_dim * kl_mask).mean(), kl_per_dim


# note: beta is not kept constant throughout training
def vae_loss_function(labels, logits, mu, logvar, beta, field2idx, dim_target_kl=0.5):
  # L = L_rec + beta*L_kl
  rec_loss, field_losses = reconstruction_loss(labels, logits, field2idx)
  kl_loss_value, kl_per_dim = kl_loss(mu, logvar, dim_target_kl)
  total_loss = rec_loss + beta*kl_loss_value
  return total_loss, rec_loss, kl_loss_value, kl_per_dim, field_losses

def annealing_beta(step, total_steps, max_beta=1.0, ratio_zero=0.5, ratio_increase=0.25):
  # ratio_zero determines the threshold below which beta is 0, ratio_increase determine the increase above ratio_zero at which beta is 1
  progress = step / total_steps

  if progress < ratio_zero:
    return 0.0
  elif progress < ratio_zero + ratio_increase:
    return ((progress - ratio_zero) / ratio_increase) * max_beta
  else:
    return max_beta

# Create Comet experiment to track our training run

def create_experiment(params, COMET_API_KEY):
  # initiate the comet experiment for tracking
  experiment = comet_ml.Experiment(api_key=COMET_API_KEY, project_name="MahlerGPT-v3")

  # log our hyperparameters, defined above, to the experiment
  for param, value in params.items():
    experiment.log_parameter(param, value)
  experiment.flush() # log the parameters promptly

  return experiment

# Define training step and validation loss computation
def train_step(x, y, step, total_steps, model, optimizer, scaler, field2idx, max_beta=1.0, ratio_zero=0.5, ratio_increase=0.25):
  # set model to train
  model.train()

  # clear old gradients
  optimizer.zero_grad(set_to_none=True)

  # mixed-precision training
  with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    logits, mu, logvar = model(x)
    beta = annealing_beta(step, total_steps, max_beta, ratio_zero, ratio_increase)
    total_loss, rec_loss, kl_loss_value, kl_per_dim, field_losses = vae_loss_function(y, logits, mu, logvar, beta, field2idx)

  # backprop (first multiples loss by a large amount)
  scaler.scale(total_loss).backward()
  # unscale the gradients
  scaler.unscale_(optimizer)
  # clip the gradients
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  # update model parameters
  scaler.step(optimizer)

  scaler.update()

  return total_loss, rec_loss, kl_loss_value, kl_per_dim, field_losses

def compute_validation_loss(validation_set, num_batches, beta, model, params, device, field2idx):
  model.eval()

  field_loss_sums = {field: 0.0 for field in token_attributes}
  field_valid_counts = {field: 0 for field in token_attributes}
  kl_sum = 0.0
  kl_per_dim_sum = torch.zeros(params["latent_dim"])
  num_examples = 0

  with torch.inference_mode():
    for _ in range(num_batches):
      x_batch, labels = get_batch(validation_set, params["seq_length"], params["batch_size"])
      x_batch, labels = x_batch.to(device), labels.to(device)

      with torch.amp.autocast(device_type="cuda", enabled=False):
        mu, logvar = model.encode(x_batch)
        logits = model.decode(x_batch, mu)

      for i, field in enumerate(token_attributes):
        field_labels = labels[:, :, i]
        field_logits = logits[field]
        flattened_labels = field_labels.reshape(-1)
        flattened_logits = field_logits.reshape(-1, field_logits.size(-1))
        field_loss = cross_entropy(flattened_logits, flattened_labels).reshape(field_labels.shape)
        ignore_idx = field2idx[field]["IGNORE"]
        valid_mask = (field_labels != ignore_idx) & (field_labels != 0)
        masked_field_loss = field_loss[valid_mask]

        if masked_field_loss.numel() > 0:
          field_loss_sums[field] += masked_field_loss.sum().item()
          field_valid_counts[field] += masked_field_loss.numel()

      kl_loss_value, kl_per_dim = kl_loss(mu, logvar)
      kl_sum += kl_loss_value.item() * mu.size(0)
      kl_per_dim_sum += kl_per_dim.float().cpu() * mu.size(0)
      num_examples += mu.size(0)

  validation_field_losses = {
      field: field_loss_sums[field] / field_valid_counts[field]
      for field in token_attributes
      if field_valid_counts[field] > 0
  }
  active_weight_sum = sum(field_weights[field] for field in validation_field_losses)
  validation_reconstruction_loss = sum(
      validation_field_losses[field] * field_weights[field]
      for field in validation_field_losses
  ) / active_weight_sum
  validation_kl_loss = kl_sum / num_examples
  validation_kl_per_dim = kl_per_dim_sum / num_examples
  validation_total_loss = validation_reconstruction_loss + beta * validation_kl_loss

  model.train()

  return (
      validation_total_loss,
      validation_reconstruction_loss,
      validation_kl_loss,
      validation_kl_per_dim,
      validation_field_losses
  )

# Plotting code
def moving_average(values, window_size):
  if len(values) < window_size: return np.array(values)
  return np.convolve(values, np.ones(window_size) / window_size, mode="valid")

def plot_losses(training_history, validation_history, validation_iterations, current_iteration, training_smoothing_window, validation_smoothing_window):
  # Display moving averages of both losses while training
  ipythondisplay.clear_output(wait=True)
  plt.figure(figsize=(11, 5))

  smoothed_training = moving_average(training_history, training_smoothing_window)
  training_offset = training_smoothing_window - 1 if len(training_history) >= training_smoothing_window else 0
  training_iterations = np.arange(len(smoothed_training)) + training_offset
  plt.plot(training_iterations, smoothed_training, label="Training loss")

  if validation_history:
    smoothed_validation = moving_average(validation_history, validation_smoothing_window)
    validation_offset = validation_smoothing_window - 1 if len(validation_history) >= validation_smoothing_window else 0
    plt.plot(validation_iterations[validation_offset:], smoothed_validation, label="Validation loss")

  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.title(f"Training and validation loss — iteration {current_iteration}")
  plt.yscale("log")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.show()
  plt.close()

def train(
    params,
    model,
    checkpoint,
    optimizer,
    scaler,
    vectorized_songs,
    validation_set,
    field2idx,
    experiment,
    device
):
  
  device = torch.device(device)

  # Default: start from scratch
  start_iter = 0
  best_validation_loss = float("inf")
  latest_validation_loss = float("inf")

  training_history = []
  validation_history = []
  validation_iterations = []

  if checkpoint is not None:
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_iter = checkpoint["iteration"] + 1
    best_validation_loss = checkpoint.get("best_validation_loss", float("inf"))
    latest_validation_loss = checkpoint.get("validation_loss", float("inf"))
    training_history = checkpoint.get("training_history", [])
    validation_history = checkpoint.get("validation_history", [])
    validation_iterations = checkpoint.get("validation_iterations", [])

    print(f"Resuming from iteration {start_iter}, training loss = {checkpoint['training_loss']}")

  # Filter out songs that are too short for the sequence length
  training_vectorized_songs = [s for s in vectorized_songs if s.shape[0] > params["seq_length"]]
  if not training_vectorized_songs:
      print(f"Warning: No training songs found that are longer than sequence length ({params['seq_length']}).")
      print("Aborting training due to insufficient data for the given sequence length.")
      exit()

  filtered_validation_set = [s for s in validation_set if s.shape[0] > params["seq_length"]]
  if not filtered_validation_set:
      print(f"Warning: No validation songs found that are longer than sequence length ({params['seq_length']}).")
      print("Aborting training due to insufficient data for the given sequence length.")
      exit()

  validation_interval = 500
  validation_batches = 7 # average this many random validation batches to reduce noise
  plot_interval = 100
  training_smoothing_window = 100
  validation_smoothing_window = 3

  training_loss = float("inf")
  last_iter = start_iter - 1
  total_steps = params["epochs"]*params["num_training_iterations"]

  if hasattr(tqdm, '_instances'): tqdm._instances.clear()
  for iter in tqdm(range(start_iter, total_steps)):
    last_iter = iter

    # Use the filtered training songs
    x_batch, y_batch = get_batch(training_vectorized_songs, params["seq_length"], params["batch_size"])
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    total_loss, rec_loss, kl_loss_value, kl_per_dim, field_losses = train_step(x_batch, y_batch, iter, total_steps, model, optimizer, scaler, field2idx, max_beta=params["max_beta"], ratio_zero=params["ratio_zero"], ratio_increase=params["ratio_increase"])

    # Logging
    training_loss = total_loss.item()
    training_history.append(training_loss)
    beta = annealing_beta(iter, total_steps, max_beta=params["max_beta"], ratio_zero=params["ratio_zero"], ratio_increase=params["ratio_increase"])

    experiment.log_metric("training_loss", training_loss, step=iter)
    experiment.log_metric("reconstruction_loss", rec_loss.item(), step=iter)
    experiment.log_metric("kl_loss", kl_loss_value.item(), step=iter)
    experiment.log_metric("beta", beta, step=iter)
    experiment.log_metric("kl/raw_mean", kl_per_dim.mean().item(), step=iter)
    experiment.log_metric("kl/active_dimensions", (kl_per_dim > 0.01).sum().item(), step=iter)

    for field, field_loss in field_losses.items():
      experiment.log_metric(f"loss/train_{field}", field_loss.item(), step=iter)

    # Evaluate validation loss every validation interval and at the very end of training
    if iter % validation_interval == 0 or iter == total_steps - 1:
      beta = annealing_beta(iter, total_steps,max_beta=params["max_beta"], ratio_zero=params["ratio_zero"], ratio_increase=params["ratio_increase"])

      # Use the filtered validation set
      latest_validation_loss, validation_rec_loss, validation_kl_loss, validation_kl_per_dim, validation_field_losses = compute_validation_loss(
          filtered_validation_set,
          validation_batches,
          beta,
          model,
          params,
          device,
          field2idx
      )

      validation_history.append(latest_validation_loss)
      validation_iterations.append(iter)

      # Log the validation loss to the Comet interface
      validation_metrics = {
          "validation_loss": latest_validation_loss,
          "loss/validation_reconstruction": validation_rec_loss,
          "loss/validation_kl": validation_kl_loss,
          "kl/validation_raw_mean": validation_kl_per_dim.mean().item(),
          "kl/validation_active_dimensions": (validation_kl_per_dim > 0.01).sum().item(),
      }

      for field, field_loss in validation_field_losses.items():
        validation_metrics[f"loss/validation_{field}"] = field_loss

      experiment.log_metrics(validation_metrics, step=iter)
      experiment.flush()

      if latest_validation_loss < best_validation_loss:
        best_validation_loss = latest_validation_loss

        best_checkpoint_data = {
            "iteration": iter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_loss": training_loss,
            "validation_loss": latest_validation_loss,
            "best_validation_loss": best_validation_loss,
            "training_history": training_history,
            "validation_history": validation_history,
            "validation_iterations": validation_iterations,
        }

        # Save the checkpoint with the best validation loss
        torch.save(best_checkpoint_data, best_checkpoint_prefix)

    if iter % plot_interval == 0 or iter == total_steps - 1:
      plot_losses(training_history, validation_history, validation_iterations, iter, training_smoothing_window, validation_smoothing_window)

    # Save model checkpoint for every multiple of 100
    if iter % 100 == 0:
      checkpoint_data = {
          "iteration": iter,
          "model_state_dict": model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
          "training_loss": training_loss,
          "validation_loss": latest_validation_loss,
          "best_validation_loss": best_validation_loss,
          "training_history": training_history,
          "validation_history": validation_history,
          "validation_iterations": validation_iterations,
      }

      torch.save(checkpoint_data, checkpoint_prefix)

  # Save the final trained model
  if last_iter >= start_iter:
    final_checkpoint_data = {
        "iteration": last_iter,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_loss": training_loss,
        "validation_loss": latest_validation_loss,
        "best_validation_loss": best_validation_loss,
        "training_history": training_history,
        "validation_history": validation_history,
        "validation_iterations": validation_iterations,
    }

    torch.save(final_checkpoint_data, checkpoint_prefix)

  experiment.flush()
  experiment.end()