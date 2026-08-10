### TOKENIZE ###

import mido
from mido import MidiFile, merge_tracks
from mido.midifiles.meta import KeySignatureError

import itertools
from pathlib import Path
from dataclasses import dataclass


"""
A compound token has:
Family: {Metric, Note, Controller, BOS, EOS, PAD)}
Bar/Position
Channel
Program
Pitch
Velocity
Duration
Tempo
Time Signature
Controller Type
Controller Value
"""

@dataclass(frozen=True)
class Token:
  Family: str
  bar_position: int | str | None # Only metric tokens use the position field!
  channel: int | str
  program: int | str
  pitch: int | str
  velocity: int | str
  duration: int | str
  tempo: int | str
  time_signature: str
  control: int | str
  control_value: int | str

@dataclass(frozen=True)
class Note:
  duration: int
  channel: int
  program: int
  absolute_position: int
  pitch: int
  velocity: int

@dataclass(frozen=True)
class TempoEvent:
  absolute_position: int
  tempo: int

@dataclass(frozen=True)
class TimeSignatureEvent:
  absolute_position: int
  numerator: int
  denominator: int

@dataclass(frozen=True)
class ControllerEvent:
  absolute_position: int
  channel: int
  control: int
  control_value: int

def midi2token (midifile_path, text_path, positions_per_quarternote=12):

  mid = MidiFile(midifile_path)

  ticks_per_quarternote = mid.ticks_per_beat

  position_per_ticks = positions_per_quarternote / ticks_per_quarternote # to get absolute_position -> round(absolute_time * position_per_ticks)

  notes, metric_events, controller_events = extract_events(mid, position_per_ticks) # returns an array of events of type Note, TempEvent, TimeSignatureEvent, and ControllerEvent

  timeline = merge_events(notes, metric_events, controller_events) # returns a single array containing arrays of events, grouped by absolute_position

  tokens = timeline2token(timeline, positions_per_quarternote)

  with open(text_path, "w") as f:
    for token in tokens:
      f.write(f"{token.Family} {token.bar_position} {token.channel} {token.program} {token.pitch} {token.velocity} {token.duration} {token.tempo} {token.time_signature} {token.control} {token.control_value}\n")

# Here we assume no overlapping notes
def extract_events(mid, position_per_ticks, print_debug=False):
  note_on_dict = dict()

  current_programs = [0]*16 # index by channel, there are 16 channels

  absolute_time = 0

  orphan_note_off = 0
  duration_zero_notes = 0
  overlapping_notes = 0
  threshold = 20000

  notes = []
  metric_events = []
  controller_events = []

  for msg in merge_tracks(mid.tracks):
    absolute_time += msg.time
    absolute_position = round(absolute_time * position_per_ticks)

    # Find note pairs
    if msg.type == "note_on" and msg.velocity > 0:
      key = (msg.note, msg.channel) # mutable objects cannot be dictionary keys! Must be a tuple.

      if key in note_on_dict:
        if print_debug: print("Detected an overlapping note!")
        overlapping_notes += 1
      else:
        note_on_dict[key] = [msg, absolute_time, current_programs[msg.channel]]

    elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0): # search for the note_on pair in note_on_queue
      key = (msg.note, msg.channel)

      if key in note_on_dict:
        note_on = note_on_dict[key] # returns [msg, absolute_time, current_programs[msg.channel]]
        note_off = [msg, absolute_time]
        del note_on_dict[key]

        # Append note to note list
        start = round(note_on[1] * position_per_ticks)
        end = round(note_off[1] * position_per_ticks) # same as absolute_position
        duration = end - start

        if duration >= 1:
          note = Note(
              duration = duration,
              channel = note_on[0].channel,
              program = note_on[2],
              absolute_position = start,
              pitch = note_on[0].note,
              velocity = note_on[0].velocity // 4 # quantize velocity to 32 bins
        )
          notes.append(note)

        else:
          if print_debug: print("Detected note of duration 0, skipping...")
          duration_zero_notes += 1

      else:
        if print_debug: print(f"Note_off msg detected but no corresponding note_on pair, rejecting...")
        orphan_note_off += 1
        if orphan_note_off > threshold:
          raise ValueError("Number of orphan note_off messages exceeds the allowed threshold.")

    elif msg.type == "program_change":
      current_programs[msg.channel] = msg.program

    # Other events
    elif msg.type == "set_tempo":
      # MIDI stores microseconds per quarter note. This converts it to
      # quarter notes per minute without depending on the time signature.
      qpm = round(60_000_000 / msg.tempo)
      tempo_event = TempoEvent(
          absolute_position=absolute_position,
          tempo=qpm,
      )
      metric_events.append(tempo_event)

    elif msg.type == "time_signature":
      time_signature_event = TimeSignatureEvent(
          absolute_position = absolute_position,
          numerator = msg.numerator,
          denominator = msg.denominator
      )
      metric_events.append(time_signature_event)

    elif msg.type == "control_change":
      control_change_event = ControllerEvent(
          absolute_position = absolute_position,
          channel = msg.channel,
          control = msg.control,
          control_value = msg.value if msg.control in (0, 32) else msg.value // 4 # preserve bank select exactly
      )
      controller_events.append(control_change_event)

  if print_debug: print(f"Orphan note_off messages: {orphan_note_off}")
  if print_debug: print(f"Duration 0 notes: {duration_zero_notes}")
  if print_debug: print(f"Overlapping notes: {overlapping_notes}")
  if print_debug: print(f"Total notes: {len(notes)}")
  print(f"Total malformed notes:{(orphan_note_off + duration_zero_notes + overlapping_notes)/len(notes)}")
  if (orphan_note_off + duration_zero_notes + overlapping_notes)/len(notes) > 0.09:
    print(mid.filename)
  return notes, metric_events, controller_events

# Group all quantized events by absolute position
def merge_events(notes, metric_events, controller_events):
  sorted_events = sorted(notes + metric_events + controller_events, key = lambda obj: obj.absolute_position) # lambda expression: lambda parameter:expression, the key parameter of sorted tells it what to sort


  return [ (position, list(group)) for position, group in itertools.groupby(sorted_events,key=lambda event: event.absolute_position)]

# NOTE
def emit_note_token(note, tokens):
    note_event = Token(
      Family = "Note",
      bar_position = "IGNORE",
      channel = note.channel,
      program = note.program,
      pitch = note.pitch,
      velocity = note.velocity,
      duration = note.duration,
      tempo = "IGNORE",
      time_signature = "IGNORE",
      control = "IGNORE",
      control_value = "IGNORE"
    )

    tokens.append(note_event)

# METRICS
def emit_position_token(absolute_position, current_bar_start, tokens, tempo_event=None):
    controller_event = Token(
        Family="Metric",
        bar_position=absolute_position - current_bar_start,
        channel="IGNORE",
        program="IGNORE",
        pitch="IGNORE",
        velocity="IGNORE",
        duration="IGNORE",
        tempo=(
            tempo_event.tempo
            if tempo_event is not None
            else "CONTINUE"
        ),
        time_signature="CONTINUE",
        control="IGNORE",
        control_value="IGNORE"
    )

    tokens.append(controller_event)

def emit_bar_token(tokens, event=None): # event=None means continue
  bar_event = Token(
      Family = "Metric",
      bar_position = "BAR",
      channel = "IGNORE",
      program = "IGNORE",
      pitch = "IGNORE",
      velocity = "IGNORE",
      duration = "IGNORE",
      tempo = "CONTINUE",
      time_signature = f"{event.numerator}/{event.denominator}" if isinstance(event, TimeSignatureEvent) else "CONTINUE",
      control = "IGNORE",
      control_value = "IGNORE"

  )

  tokens.append(bar_event)

# CONTROLLER
def emit_controller_token(event, tokens):
  controller_event = Token(
      Family = "Controller",
      bar_position = "IGNORE",
      channel = event.channel,
      program = "IGNORE",
      pitch = "IGNORE",
      velocity = "IGNORE",
      duration = "IGNORE",
      tempo = "IGNORE",
      time_signature = "IGNORE",
      control = event.control,
      control_value = event.control_value
  )

  tokens.append(controller_event)

# Computes bar_length
def bar_length(time_signature_nn, time_signature_dd, positions_per_quarternote):
      length = round(time_signature_nn * 4 * positions_per_quarternote / time_signature_dd)
      return max(1, length)


# Generate BOS, BAR, POSITION, event tokens, EOS
def timeline2token(timeline, positions_per_quarternote):
    tokens = []

    # Emit BOS token
    BOS = Token(
        Family = "BOS",
        bar_position = "IGNORE",
        channel = "IGNORE",
        program  = "IGNORE",
        pitch = "IGNORE",
        velocity = "IGNORE",
        duration = "IGNORE",
        tempo = "IGNORE",
        time_signature = "IGNORE",
        control = "IGNORE",
        control_value = "IGNORE"
    )

    tokens.append(BOS)

    # Default
    current_tempo = 120
    current_time_signature_nn = 4
    current_time_signature_dd = 4

    current_absolute_position = 0
    current_bar_start = 0

    # extract initial events

    initial_events = ( timeline[0][1] if timeline and timeline[0][0] == 0 else [] )

    initial_time_signatures = [ event for event in initial_events if isinstance(event, TimeSignatureEvent) ]

    initial_time_signature = ( initial_time_signatures[-1] if initial_time_signatures else TimeSignatureEvent(0, 4, 4) )

    current_time_signature_nn = initial_time_signature.numerator
    current_time_signature_dd = initial_time_signature.denominator

    emit_bar_token(tokens, initial_time_signature)

    next_bar_start = current_bar_start + bar_length(current_time_signature_nn, current_time_signature_dd, positions_per_quarternote)

    # Iterate over events

    for absolute_position, events in timeline:
      # Find changes once for this whole position group. If malformed MIDI
      # contains several at the same position, the last one wins.
      time_signature_events = [ event for event in events if isinstance(event, TimeSignatureEvent) ]

      time_signature_change = time_signature_events[-1] if time_signature_events else None

      tempo_events = [ event for event in events if isinstance(event, TempoEvent) ]

      tempo_change = tempo_events[-1] if tempo_events else None

      # Emit any complete empty bars crossed before this event position.
      while next_bar_start < absolute_position:
          current_bar_start = next_bar_start
          emit_bar_token(tokens)
          next_bar_start = current_bar_start + bar_length(current_time_signature_nn, current_time_signature_dd, positions_per_quarternote)

      # The event lies exactly on the next regular bar boundary.
      if absolute_position == next_bar_start:
          current_bar_start = absolute_position

          if time_signature_change is not None:
              current_time_signature_nn = time_signature_change.numerator

              current_time_signature_dd = time_signature_change.denominator
              
              emit_bar_token(tokens, time_signature_change)
          else:
              emit_bar_token(tokens)

          next_bar_start = current_bar_start + bar_length(current_time_signature_nn, current_time_signature_dd, positions_per_quarternote)

      # A time-signature change inside the current bar ends that bar early
      # and begins a new bar at the change position.
      elif time_signature_change is not None and absolute_position > current_bar_start:
          current_bar_start = absolute_position
          current_time_signature_nn = time_signature_change.numerator
          current_time_signature_dd = time_signature_change.denominator

          emit_bar_token(tokens, time_signature_change)
          next_bar_start = current_bar_start + bar_length(current_time_signature_nn, current_time_signature_dd, positions_per_quarternote)

      # Exactly one POSITION token is emitted for the whole event group.
      emit_position_token(absolute_position,current_bar_start, tokens, tempo_change)

      # Deterministic ordering for simultaneous events.
      controller_events = sorted( (event for event in events if isinstance(event, ControllerEvent)), key=lambda event: (event.channel, event.control))
      note_events = sorted((event for event in events if isinstance(event, Note)), key=lambda event: (event.channel, event.program, event.pitch))

      for event in controller_events:
          emit_controller_token(event, tokens)

      for event in note_events:
          emit_note_token(event, tokens)

    # Emit EOS token
    tokens.append(
        Token(
            Family="EOS",
            bar_position="IGNORE",
            channel="IGNORE",
            program="IGNORE",
            pitch="IGNORE",
            velocity="IGNORE",
            duration="IGNORE",
            tempo="IGNORE",
            time_signature="IGNORE",
            control="IGNORE",
            control_value="IGNORE",
        )
    )

    return tokens