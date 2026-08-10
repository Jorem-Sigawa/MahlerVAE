### Use this to sample specific bars from a tokenized prompt file ###

import os
from pathlib import Path

def sample_bars(token_text_path, out_path, bar_start, bar_end):
  with open(token_text_path, "r") as tfile:
    with open(out_path, "w") as ofile:
      bar_count = 0
      for line in tfile:
        line = line.strip()
        compound_token = line.split()

        if compound_token[1] == "BAR":
          bar_count += 1

        if bar_count >= bar_start and bar_count <= bar_end:
          ofile.write(line + "\n")
        elif bar_count > bar_end:
          break

# Example usage:
token_text_path = Path("PUT YOUR TOKENIZED PROMPT FILE PATH HERE") #e.g., /datasets/CPRemi/Training-Set/mahlers-8th-symphony-finale.txt

mid_path = "PUT YOUR OUTPUT DIRECTORY PATH HERE" # e.g., /sampled_bars/CPRemi/

os.makedirs(mid_path, exist_ok=True)
bar_start = 29
bar_end = 44
out_path = mid_path + f"/{token_text_path.stem}f{bar_start}t{bar_end}.txt"
sample_bars(token_text_path, out_path, bar_start, bar_end)