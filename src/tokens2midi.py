### TOKENS2MIDI ###

import mido
from mido import (
    MidiFile,
    merge_tracks,
    MidiTrack,
    tempo2bpm,
    bpm2tempo )
from mido.midifiles.meta import KeySignatureError

import itertools
from pathlib import Path
from dataclasses import dataclass

# Example of a token in text:
# Note IGNORE 9 1 38 27 6 IGNORE IGNORE IGNORE IGNORE
# Metric BAR IGNORE IGNORE IGNORE IGNORE IGNORE IGNORE 4/4 IGNORE IGNORE

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

# PRIORITY
# TIME_SIGNATURE_PRIORITY = 0
# TEMPO_PRIORITY = 1
# PROGRAM_CHANGE_PRIORITY = 2
# CONTROL_CHANGE_PRIORITY = 3
# NOTE_OFF_PRIORITY = 4
# NOTE_ON_PRIORITY = 5

# Computes bar_length
def bar_length(time_signature_nn, time_signature_dd, positions_per_quarternote):
      length = round(time_signature_nn * 4 * positions_per_quarternote / time_signature_dd)
      return max(1, length)

def get_int_value(token_value, default_value):
    if token_value == "IGNORE" or token_value == "CONTINUE":
        return default_value
    try:
        return int(token_value)
    except ValueError:
        return default_value # Fallback for unexpected non-int value

def get_float_value(token_value, default_value):
    if token_value == "IGNORE" or token_value == "CONTINUE":
        return default_value
    try:
        return float(token_value)
    except ValueError:
        return default_value # Fallback for unexpected non-float value

def token2midi(token_text_path, out_path, ticks_per_beat=480, positions_per_beat=12):
  midi_msgs = []

  # Number of MIDI ticks represented by one REMI position.
  ticks_per_position = ticks_per_beat / positions_per_beat

  current_programs = [0]*16 # one for each channel

  # Default
  current_numerator = 4
  current_denominator = 4

  bar_start_tick = 0
  current_tick = 0
  encountered_first_bar = False

  with open(token_text_path, "r") as file:
    for line in file:
      line = line.strip()
      compound_token = line.split()

      family = compound_token[0]

      # Process by family
      if family == "Note":

        channel = get_int_value(compound_token[2], 0)
        program = get_int_value(compound_token[3], 0)

        if program != current_programs[channel]:
          # send a program_change msg first
          midi_msgs.append([current_tick, 2, (mido.Message("program_change", channel = channel, program = program))])
          current_programs[channel] = program

        pitch = get_int_value(compound_token[4], 60) # Default to middle C
        velocity = min(127, get_int_value(compound_token[5], 64)*4 + 2) # Default velocity 64
        duration = get_int_value(compound_token[6], 1) # this is in position units, default to 1

        # send a note_on and a note_off token to midi_msgs
        duration_ticks = round(duration*ticks_per_position)
        absolute_time_off = current_tick + duration_ticks

        midi_msgs.append([current_tick, 5, mido.Message("note_on", note=pitch, velocity=velocity, channel=channel)])

        midi_msgs.append([absolute_time_off, 4, mido.Message("note_off", note=pitch, velocity=0, channel=channel)])

      elif family == "Metric": # encodes bar, position, time_signature, and tempo events

        if compound_token[1] == "BAR": # encodes bar and time_signature events

          # tick update
          if encountered_first_bar:
            positions_per_bar = bar_length(current_numerator, current_denominator, positions_per_beat)
            bar_start_tick += round(positions_per_bar*ticks_per_position)

          else:
            encountered_first_bar = True

          current_tick = bar_start_tick

          time_signature = compound_token[8]
          if time_signature != "CONTINUE":
            if time_signature == "IGNORE": # Default time signature
              num = 4
              den = 4
            else:
              try:
                num = int(time_signature.split("/")[0])
                den = int(time_signature.split("/")[1])
              except ValueError:
                num = 4 # Fallback for malformed time signature
                den = 4
            #send a time_signature event message
            midi_msgs.append([current_tick, 0, (mido.MetaMessage("time_signature", numerator=num, denominator=den))])

            current_numerator = num
            current_denominator = den

        elif compound_token[1] != "BAR": # encodes position and tempo events
          position = get_int_value(compound_token[1], 0)

          current_tick = round(bar_start_tick + position*ticks_per_position)

          bpm_str = compound_token[7]
          if bpm_str != "CONTINUE":
            bpm = get_float_value(bpm_str, 120.0) # Default tempo 120 bpm
            # send a set_tempo message
            midi_msgs.append([current_tick, 1, (mido.MetaMessage("set_tempo", tempo=bpm2tempo(bpm)))])

      elif family == "Controller":
        channel = get_int_value(compound_token[2], 0)
        controller_type = get_int_value(compound_token[9], 0)
        controller_value = get_int_value(compound_token[10], 0)
        if controller_type not in (0, 32): controller_value = min(127, controller_value*4 + 2)

        # send a control_change message
        priority = 1.5 if controller_type in (0, 32) else 3
        midi_msgs.append([current_tick, priority, (mido.Message("control_change", channel=channel, control=controller_type, value=controller_value))])

    # Convert midi_msgs from absolute time to delta time
    midi_msgs.sort(key=lambda event: (event[0], event[1]))

    track = MidiTrack()
    previous_absolute_tick = 0

    for absolute_tick, priority, message in midi_msgs:
      delta_tick = absolute_tick - previous_absolute_tick

      # All in one track, .copy allows us to copy message and update a parameter
      track.append(message.copy(time=int(delta_tick)))

      previous_absolute_tick = absolute_tick

  mid = MidiFile(ticks_per_beat=ticks_per_beat)
  mid.tracks.append(track)
  mid.save(out_path)
