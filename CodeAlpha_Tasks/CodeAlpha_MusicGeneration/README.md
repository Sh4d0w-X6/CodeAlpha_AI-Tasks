# 🎵 Task 3 — Music Generation with AI
**CodeAlpha AI Internship**

## Features
- **Markov Chain** generator — instant music, no training needed
- **LSTM model** training script for deep-learning based generation
- Multiple musical scales (C Major, A Minor, G Major, Blues, Pentatonic, etc.)
- Adjustable tempo (BPM) and note count
- Save output as **MIDI** (.mid) or **MusicXML** (.xml)
- Dark-themed GUI with generation log

## Setup & Run

### Quick Start (Markov Chain Generator)
```bash
pip install -r requirements.txt
python music_gen.py
```

### Full LSTM Training
```bash
# With your own MIDI files:
python train_lstm.py --midi_dir ./midi_data --epochs 50

# Without MIDI files (uses music21 Bach corpus):
python train_lstm.py
```

## How It Works

### Markov Chain (Instant)
1. Define note transitions based on the selected scale
2. Walk the transition graph randomly with rhythmic variation
3. Export to MIDI

### LSTM (Deep Learning)
1. Parse MIDI files using `music21`
2. Build integer-encoded note sequences
3. Train LSTM (256 units × 2 layers) to predict next notes
4. Generate new sequences using temperature-scaled sampling
5. Convert output back to MIDI

## Tech Stack
- **Python** + **Tkinter** (GUI)
- **music21** (MIDI parsing and export)
- **TensorFlow / Keras** (LSTM model)
- **NumPy** (sequence processing)

## File Structure
```
Task3_MusicGeneration/
├── music_gen.py      # Main GUI app (Markov generation)
├── train_lstm.py     # LSTM training script
├── requirements.txt
└── README.md
```
