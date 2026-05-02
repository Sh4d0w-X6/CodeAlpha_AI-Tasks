"""
CodeAlpha Internship - Task 3: LSTM Training Script
Run this separately to train an LSTM on MIDI data.
Usage: python train_lstm.py --midi_dir ./midi_data --epochs 50
"""

import os
import argparse
import pickle
import numpy as np

def install_if_missing(pkg, import_name=None):
    import importlib, subprocess, sys
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_if_missing("music21")
install_if_missing("tensorflow")
install_if_missing("numpy")

from music21 import corpus, converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def extract_notes_from_midi(midi_path):
    """Extract notes and chords from a MIDI file."""
    notes = []
    try:
        midi = converter.parse(midi_path)
        parts = instrument.partitionByInstrument(midi)
        elements = parts.parts[0] if parts else midi.flat
        for el in elements.flat.notesAndRests:
            if isinstance(el, note.Note):
                notes.append(str(el.pitch))
            elif isinstance(el, chord.Chord):
                notes.append(".".join(str(n) for n in el.normalOrder))
    except Exception as e:
        print(f"  Warning: could not parse {midi_path}: {e}")
    return notes


def load_music21_corpus_samples():
    """Load sample MIDI data from music21's built-in corpus."""
    print("Loading music21 corpus samples (Bach chorales)...")
    all_notes = []
    # music21 has Bach chorales built in
    paths = corpus.getComposer('bach')[:10]  # use first 10
    for p in paths:
        try:
            score = corpus.parse(p)
            for part in score.parts[:1]:  # first part only
                for el in part.flat.notesAndRests:
                    if isinstance(el, note.Note):
                        all_notes.append(str(el.pitch))
                    elif isinstance(el, chord.Chord):
                        all_notes.append(".".join(str(n) for n in el.normalOrder))
        except Exception as e:
            print(f"  Skip {p}: {e}")
    return all_notes


def train(midi_dir=None, epochs=50, seq_length=50, output_dir="./model_output"):
    os.makedirs(output_dir, exist_ok=True)

    # Load notes
    if midi_dir and os.path.isdir(midi_dir):
        print(f"Loading MIDI files from {midi_dir}...")
        all_notes = []
        for f in os.listdir(midi_dir):
            if f.endswith((".mid", ".midi")):
                print(f"  Parsing: {f}")
                all_notes.extend(extract_notes_from_midi(os.path.join(midi_dir, f)))
    else:
        print("No MIDI directory provided — using music21 Bach corpus...")
        all_notes = load_music21_corpus_samples()

    if len(all_notes) < seq_length + 10:
        print("❌ Not enough notes to train. Provide more MIDI files.")
        return

    print(f"\n✅ Total notes extracted: {len(all_notes)}")
    unique = sorted(set(all_notes))
    n_vocab = len(unique)
    print(f"   Unique elements: {n_vocab}")

    note_to_int = {n: i for i, n in enumerate(unique)}
    int_to_note = {i: n for n, i in note_to_int.items()}

    # Save mappings
    with open(os.path.join(output_dir, "mappings.pkl"), "wb") as f:
        pickle.dump({"note_to_int": note_to_int, "int_to_note": int_to_note, "unique": unique}, f)

    # Prepare sequences
    X, y = [], []
    for i in range(len(all_notes) - seq_length):
        X.append([note_to_int[n] for n in all_notes[i:i + seq_length]])
        y.append(note_to_int[all_notes[i + seq_length]])

    X = np.reshape(X, (len(X), seq_length, 1)) / float(n_vocab)
    y = to_categorical(y, num_classes=n_vocab)

    print(f"   Training samples: {len(X)}")

    # Build LSTM model
    model = Sequential([
        LSTM(256, input_shape=(seq_length, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(n_vocab, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    checkpoint_path = os.path.join(output_dir, "model_best.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='loss', verbose=1),
        EarlyStopping(patience=10, restore_best_weights=True)
    ]

    print(f"\n🚀 Training for up to {epochs} epochs...")
    model.fit(X, y, epochs=epochs, batch_size=64, callbacks=callbacks)
    print(f"\n✅ Model saved to {checkpoint_path}")
    print(f"   Mappings saved to {output_dir}/mappings.pkl")
    print("\nTo generate music, run music_gen.py and load the trained model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM on MIDI data")
    parser.add_argument("--midi_dir", default=None, help="Path to folder with .mid files")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seq_length", type=int, default=50)
    parser.add_argument("--output_dir", default="./model_output")
    args = parser.parse_args()
    train(args.midi_dir, args.epochs, args.seq_length, args.output_dir)
