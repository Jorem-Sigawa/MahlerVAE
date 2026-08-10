import os
import torch

def load_checkpoint(model, checkpoint_dir, device, use_best_ckpt=True):
  ### Load previously saved checkpoint ###
  os.makedirs(checkpoint_dir, exist_ok=True)
  checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt.pt")
  best_checkpoint_prefix = os.path.join(checkpoint_dir, "best_ckpt.pt")

  if not use_best_ckpt:
    if os.path.exists(checkpoint_prefix):
        checkpoint = torch.load(checkpoint_prefix, map_location=device) 
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    else:
        print(f"Checkpoint not found at {checkpoint_prefix}. Using the best checkpoint instead.")
        if os.path.exists(best_checkpoint_prefix):
            checkpoint = torch.load(best_checkpoint_prefix, map_location=device) 
            model.load_state_dict(checkpoint["model_state_dict"])
            return model
        else:
            print(f"No checkpoint found at {best_checkpoint_prefix}. The model will use randomly initialized weights.")
            return model

  elif use_best_ckpt:
    if os.path.exists(best_checkpoint_prefix):
        checkpoint = torch.load(best_checkpoint_prefix, map_location=device) 
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    else:
        print(f"Best checkpoint not found at {best_checkpoint_prefix}. Using the latest checkpoint instead.")
        if os.path.exists(checkpoint_prefix):
            checkpoint = torch.load(checkpoint_prefix, map_location=device) 
            model.load_state_dict(checkpoint["model_state_dict"])
            return model
        else:
           print(f"No checkpoint found at {checkpoint_prefix}. The model will use randomly initialized weights.")
           return model
