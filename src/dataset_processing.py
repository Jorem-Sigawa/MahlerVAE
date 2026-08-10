import itertools
import numpy as np
from pathlib import Path
from tqdm import tqdm

### DATASET DOWNLOAD & INSPECTION

def dataset_processing(midi_path, validation_midi_path):
    midi_path = Path(midi_path)
    validation_midi_path = Path(validation_midi_path)

    # Token attributes by idx
    # 0: family
    # 1: bar_position
    # 2: channel
    # 3: program
    # 4: pitch
    # 5: velocity
    # 6: duration
    # 7: tempo
    # 8: time_signature
    # 9: controller_type
    # 10: controller_value

    midi_tokens = [] # shape: (N, S, A) where N is # of midi tracks in the dataset, S is the number of tokens in that track, A is # of token attributes
    for item in midi_path.iterdir():
        track_tokens = []
        with open(item, "r", encoding="latin-1") as file:
            track_tokens.extend(line.strip().split() for line in file if line.strip())
        midi_tokens.append(track_tokens)

    validation_midi_tokens = []
    for item in validation_midi_path.iterdir():
        valid_track_tokens = []
        with open(item, "r", encoding="latin-1") as file:
            valid_track_tokens.extend(line.strip().split() for line in file if line.strip())
        validation_midi_tokens.append(valid_track_tokens)

    # Construct vocabulary

    token_attributes = ["family", "bar_position", "channel", "program", "pitch", "velocity", "duration", "tempo", "time_signature", "controller_type", "controller_value"]
    special_tokens = ["PAD", "UNK", "IGNORE", "CONTINUE"] # PAD must have index zero in every field

    field_vocab = {attribute: special_tokens + sorted({token[field_index] for token in itertools.chain.from_iterable(midi_tokens)} - set(special_tokens)) for field_index, attribute in enumerate(token_attributes)} # this is a dictionary of arrays with key attribute

    field2idx = {attribute: {value: i for i, value in enumerate(field_vocab[attribute])} for attribute in token_attributes} # a dictionary of dictionaries
    idx2field = {attribute: {i: value for i, value in enumerate(field_vocab[attribute])} for attribute in token_attributes}

    vocab_sizes = {attribute: len(field_vocab[attribute]) for attribute in token_attributes}

    # Check

    print(field2idx)
    print(idx2field)
    print(vocab_sizes)

    # Vectorize all tokens
    # To avoid session crashing due to lack of memory

    def choose_integer_dtype(vocab_sizes):
        largest_id = max(vocab_sizes.values()) - 1

        if largest_id <= np.iinfo(np.uint8).max:
            return np.uint8
        elif largest_id <= np.iinfo(np.uint16).max:
            return np.uint16
        elif largest_id <= np.iinfo(np.uint32).max:
            return np.uint32
        else:
            return np.uint64

    TOKEN_DTYPE = choose_integer_dtype(vocab_sizes)
    print("Storage dtype:", TOKEN_DTYPE)

    def vectorize_compound_tokens(midi_tokens):
        num_tokens = len(midi_tokens)
        num_fields = len(token_attributes)

        # Allocate only the final array.
        vectorized = np.empty(shape=(num_tokens, num_fields), dtype=TOKEN_DTYPE)

        mappings = [field2idx[attribute] for attribute in token_attributes]
        unknown_ids = [mapping["UNK"] for mapping in mappings]

        for row_index, token in enumerate(tqdm(midi_tokens)):
            if len(token) != num_fields:
                raise ValueError(f"Token {row_index} has {len(token)} fields; expected {num_fields}.\nToken: {token}")

            for field_index, value in enumerate(token):
                vectorized[row_index, field_index] = mappings[field_index].get(value, unknown_ids[field_index])

        return vectorized

    vectorized_songs = [vectorize_compound_tokens(song) for song in midi_tokens]
    validation_set = [vectorize_compound_tokens(song) for song in validation_midi_tokens]

    return vectorized_songs, validation_set, field2idx, idx2field, vocab_sizes

