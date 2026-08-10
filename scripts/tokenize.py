 ### Use this script to tokenize your dataset! ###

from pathlib import Path
from mido import KeySignatureError
from src.midi2tokens import midi2token

# Example usage:

training_dataset_path = Path("YOUR PATH HERE") # e.g., "datasets/MIDI/SymphonyNet_Dataset"
validation_dataset_path = Path("YOUR PATH HERE")

output_training_dir = Path("YOUR PATH HERE") # e.g., "datasets/CPRemi/SymphonyNet/Training-Set"
output_validation_dir = Path("YOUR PATH HERE") # e.g., "datasets/CPRemi/SymphonyNet/Validation-Set"

answer = input("Are you sure you want to regenerate the dataset midi files:? [Y/N]")

if answer.lower() != "y":
  print("Aborting...")
  exit()

else:
  # Converting training set to tokens
  for piece in training_dataset_path.iterdir():
      if piece.suffix.lower() == ".midi" or piece.suffix.lower() == ".mid":
        if (output_training_dir / f"{piece.stem}.txt").exists():
          print(f"Skipping file {piece.name} as it has already been converted")
        else:
          try:
            midi2token(str(training_dataset_path / f"{piece}"), str(output_training_dir / f"{piece.stem}.txt"))
          except (KeySignatureError, OSError) as e:
            print(f"Skipping file {piece.name} due to error: {e}")

  # Converting validation set to tokens
  for piece in validation_dataset_path.iterdir():
    if piece.suffix.lower() == ".midi" or piece.suffix.lower() == ".mid":
      try:
        midi2token(str(validation_dataset_path / f"{piece}"), str(output_validation_dir / f"{piece.stem}.txt"))
      except (KeySignatureError, OSError) as e:
        print(f"Skipping file {piece.name} due to error: {e}")