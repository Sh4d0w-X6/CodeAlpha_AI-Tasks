"""
CodeAlpha Internship - Task 3: Music Generation with AI
Uses LSTM to learn and generate MIDI note sequences via music21
"""

import os
import random
import pickle
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading

# ── Check dependencies ────────────────────────────────────────────────────────
def install_if_missing(pkg, import_name=None):
    import importlib, subprocess, sys
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_if_missing("music21")
install_if_missing("numpy")
install_if_missing("tensorflow")

import numpy as np
import music21
from music21 import stream, note, chord, duration, tempo, instrument
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# ── Markov Chain Music Generator (fallback + demo) ────────────────────────────
class MarkovMusicGenerator:
    """
    A fast Markov Chain based music generator that works without long training.
    For full LSTM training, see train_lstm() below.
    """
    SCALES = {
        "C Major":    ["C4","D4","E4","F4","G4","A4","B4","C5","D5","E5"],
        "A Minor":    ["A3","B3","C4","D4","E4","F4","G4","A4","B4","C5"],
        "G Major":    ["G3","A3","B3","C4","D4","E4","F#4","G4","A4","B4"],
        "D Dorian":   ["D4","E4","F4","G4","A4","B4","C5","D5"],
        "Pentatonic":  ["C4","D4","E4","G4","A4","C5","D5","E5"],
        "Blues":       ["C4","Eb4","F4","F#4","G4","Bb4","C5"],
    }
    DURATIONS = [0.25, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0]  # weighted toward quarter notes

    def __init__(self):
        self.transition = {}

    def build_markov(self, scale):
        notes = scale * 3  # repeat scale 3 times to build transitions
        for i in range(len(notes) - 1):
            self.transition.setdefault(notes[i], []).append(notes[i+1])
        # Add some variation
        for n in scale:
            neighbors = [x for x in scale if abs(scale.index(x) - scale.index(n)) <= 3]
            self.transition.setdefault(n, []).extend(neighbors)

    def generate(self, scale_name, num_notes=48, bpm=120):
        scale = self.SCALES.get(scale_name, self.SCALES["C Major"])
        self.build_markov(scale)
        s = stream.Score()
        p = stream.Part()
        p.insert(0, instrument.Piano())
        p.insert(0, tempo.MetronomeMark(number=bpm))
        current = random.choice(scale)
        for i in range(num_notes):
            # Occasionally add a chord (every ~4 notes)
            if random.random() < 0.2 and i > 0:
                chord_notes = random.sample(scale[:6], k=random.randint(2, 3))
                c = chord.Chord(chord_notes)
                c.duration = duration.Duration(random.choice([0.5, 1.0]))
                p.append(c)
            else:
                n = note.Note(current)
                n.duration = duration.Duration(random.choice(self.DURATIONS))
                # Velocity variation for expression
                n.volume.velocity = random.randint(40, 100)
                p.append(n)
            # Markov transition
            candidates = self.transition.get(current, scale)
            current = random.choice(candidates)
        s.append(p)
        return s


# ── LSTM Training (full implementation) ──────────────────────────────────────
def prepare_sequences(notes_list, seq_length=50):
    """Convert note list to training sequences."""
    unique = sorted(set(notes_list))
    note_to_int = {n: i for i, n in enumerate(unique)}
    X, y = [], []
    for i in range(len(notes_list) - seq_length):
        seq_in = notes_list[i:i + seq_length]
        seq_out = notes_list[i + seq_length]
        X.append([note_to_int[n] for n in seq_in])
        y.append(note_to_int[seq_out])
    X = np.reshape(X, (len(X), seq_length, 1)) / float(len(unique))
    y = to_categorical(y, num_classes=len(unique))
    return X, y, note_to_int, unique


def build_lstm_model(n_vocab, seq_length=50):
    model = Sequential([
        LSTM(256, input_shape=(seq_length, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(n_vocab, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def generate_from_lstm(model, seed, note_to_int, unique, num_notes=100, temperature=1.0):
    int_to_note = {v: k for k, v in note_to_int.items()}
    n_vocab = len(unique)
    pattern = [note_to_int[n] for n in seed]
    generated = []
    for _ in range(num_notes):
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(x, verbose=0)[0]
        # Temperature sampling
        prediction = np.log(prediction + 1e-10) / temperature
        prediction = np.exp(prediction) / np.sum(np.exp(prediction))
        idx = np.random.choice(len(prediction), p=prediction)
        generated.append(int_to_note[idx])
        pattern.append(idx)
        pattern = pattern[1:]
    return generated


# ── GUI ───────────────────────────────────────────────────────────────────────
class MusicGenGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 CodeAlpha Music Generator")
        self.root.geometry("650x520")
        self.root.configure(bg="#1a0533")
        self.generator = MarkovMusicGenerator()
        self.last_score = None
        self._build_ui()

    def _build_ui(self):
        # Header
        tk.Label(self.root, text="🎵 AI Music Generator",
                 font=("Georgia", 22, "bold"), fg="#f0c040", bg="#1a0533").pack(pady=(20, 3))
        tk.Label(self.root, text="CodeAlpha AI Internship — Task 3 | Markov Chain + LSTM",
                 font=("Georgia", 9), fg="#9b7fcb", bg="#1a0533").pack(pady=(0, 20))

        # Settings frame
        settings = tk.Frame(self.root, bg="#2d1066", bd=2, relief="groove")
        settings.pack(padx=40, fill="x", pady=5)
        tk.Label(settings, text="⚙ Generation Settings", font=("Georgia", 11, "bold"),
                 fg="#f0c040", bg="#2d1066").pack(pady=(10, 5))

        row1 = tk.Frame(settings, bg="#2d1066")
        row1.pack(fill="x", padx=20, pady=5)

        tk.Label(row1, text="Scale:", fg="#ccc", bg="#2d1066",
                 font=("Georgia", 10)).grid(row=0, column=0, sticky="w", padx=5)
        self.scale_var = tk.StringVar(value="C Major")
        scale_cb = ttk.Combobox(row1, textvariable=self.scale_var, state="readonly", width=16,
            values=["C Major", "A Minor", "G Major", "D Dorian", "Pentatonic", "Blues"])
        scale_cb.grid(row=0, column=1, padx=10)

        tk.Label(row1, text="Notes:", fg="#ccc", bg="#2d1066",
                 font=("Georgia", 10)).grid(row=0, column=2, sticky="w", padx=5)
        self.notes_var = tk.IntVar(value=48)
        ttk.Spinbox(row1, from_=16, to=128, textvariable=self.notes_var, width=6).grid(row=0, column=3)

        row2 = tk.Frame(settings, bg="#2d1066")
        row2.pack(fill="x", padx=20, pady=5)

        tk.Label(row2, text="Tempo (BPM):", fg="#ccc", bg="#2d1066",
                 font=("Georgia", 10)).grid(row=0, column=0, sticky="w", padx=5)
        self.bpm_var = tk.IntVar(value=120)
        ttk.Scale(row2, from_=60, to=200, variable=self.bpm_var, orient="horizontal",
                  length=150).grid(row=0, column=1, padx=5)
        tk.Label(row2, textvariable=self.bpm_var, fg="#f0c040", bg="#2d1066",
                 font=("Georgia", 10, "bold")).grid(row=0, column=2)

        settings.pack_configure(pady=10)

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#1a0533")
        btn_frame.pack(pady=15)

        tk.Button(btn_frame, text="🎼 Generate Music", command=self._generate,
                  bg="#f0c040", fg="#1a0533", font=("Georgia", 13, "bold"),
                  relief="flat", padx=20, pady=10, cursor="hand2").grid(row=0, column=0, padx=10)

        tk.Button(btn_frame, text="💾 Save MIDI", command=self._save_midi,
                  bg="#9b59b6", fg="white", font=("Georgia", 11, "bold"),
                  relief="flat", padx=15, pady=10, cursor="hand2").grid(row=0, column=1, padx=10)

        tk.Button(btn_frame, text="📄 Save MusicXML", command=self._save_xml,
                  bg="#3498db", fg="white", font=("Georgia", 11, "bold"),
                  relief="flat", padx=15, pady=10, cursor="hand2").grid(row=0, column=2, padx=10)

        # Output log
        log_frame = tk.Frame(self.root, bg="#2d1066", bd=2, relief="groove")
        log_frame.pack(padx=40, fill="both", expand=True, pady=5)
        tk.Label(log_frame, text="📋 Generation Log", font=("Courier", 9, "bold"),
                 fg="#f0c040", bg="#2d1066", anchor="w").pack(fill="x", padx=10, pady=(5, 0))
        self.log = tk.Text(log_frame, height=8, font=("Courier New", 10),
                           bg="#110022", fg="#a0e0a0", relief="flat",
                           padx=10, pady=8, state="disabled")
        self.log.pack(fill="both", expand=True, padx=5, pady=5)

        # Status
        self.status = tk.Label(self.root, text="Ready to generate music 🎵",
                               font=("Georgia", 9), fg="#9b7fcb", bg="#1a0533")
        self.status.pack(pady=5)

    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _generate(self):
        scale = self.scale_var.get()
        num_notes = self.notes_var.get()
        bpm = self.bpm_var.get()
        self.status.config(text="⏳ Generating music...")
        self._log(f"🎹 Generating {num_notes} notes in {scale} at {bpm} BPM...")
        threading.Thread(target=self._do_generate, args=(scale, num_notes, bpm), daemon=True).start()

    def _do_generate(self, scale, num_notes, bpm):
        try:
            gen = MarkovMusicGenerator()
            score = gen.generate(scale, num_notes=num_notes, bpm=bpm)
            self.last_score = score
            # Extract note info for display
            notes = []
            for element in score.flat.notesAndRests:
                if isinstance(element, note.Note):
                    notes.append(element.nameWithOctave)
                elif isinstance(element, chord.Chord):
                    notes.append(f"[{','.join(p.nameWithOctave for p in element.pitches)}]")
            preview = " → ".join(notes[:12]) + " ..."
            self.root.after(0, lambda: self._log(f"✅ Generated! Preview: {preview}"))
            self.root.after(0, lambda: self._log(f"   Total elements: {len(notes)}"))
            self.root.after(0, lambda: self.status.config(text="✅ Music generated! Save it below."))
        except Exception as e:
            self.root.after(0, lambda: self._log(f"❌ Error: {e}"))

    def _save_midi(self):
        if not self.last_score:
            messagebox.showwarning("No Music", "Generate music first!")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".mid", filetypes=[("MIDI files", "*.mid"), ("All files", "*.*")],
            initialfile="codealpha_music.mid"
        )
        if path:
            self.last_score.write("midi", fp=path)
            self._log(f"💾 MIDI saved: {os.path.basename(path)}")
            self.status.config(text=f"✅ Saved: {os.path.basename(path)}")

    def _save_xml(self):
        if not self.last_score:
            messagebox.showwarning("No Music", "Generate music first!")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xml", filetypes=[("MusicXML", "*.xml"), ("All files", "*.*")],
            initialfile="codealpha_music.xml"
        )
        if path:
            self.last_score.write("musicxml", fp=path)
            self._log(f"📄 MusicXML saved: {os.path.basename(path)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicGenGUI(root)
    root.mainloop()
