# 🤖 CodeAlpha AI Internship — All 4 Tasks
> Complete implementation of all 4 AI Internship tasks by CodeAlpha

---

## 📁 Project Structure

```
CodeAlpha_Tasks/
├── Task1_LanguageTranslation/
│   ├── app.py                  # Main GUI app
│   ├── requirements.txt
│   └── README.md
│
├── Task2_ChatbotFAQ/
│   ├── chatbot.py              # NLP FAQ chatbot
│   └── README.md
│
├── Task3_MusicGeneration/
│   ├── music_gen.py            # Markov Chain GUI app
│   ├── train_lstm.py           # LSTM training script
│   ├── requirements.txt
│   └── README.md
│
└── Task4_ObjectDetection/
    ├── object_detection.py     # YOLO + Tracker app
    ├── requirements.txt
    └── README.md
```

---

## ✅ Task 1 — Language Translation Tool
| Feature | Detail |
|---------|--------|
| Languages | 20+ (Hindi, Spanish, French, Arabic, Chinese, ...) |
| API | Google Translate via `deep-translator` (free, no key) |
| Extras | Auto-detect, swap languages, copy to clipboard, TTS |
| Run | `pip install deep-translator pyttsx3` → `python app.py` |

---

## ✅ Task 2 — Chatbot for FAQs
| Feature | Detail |
|---------|--------|
| FAQs | 16 Q&A pairs about CodeAlpha internship |
| NLP | Custom TF-IDF + Cosine Similarity (pure Python) |
| Extras | Quick-suggestion buttons, dark chat UI |
| Run | `python chatbot.py` (no dependencies needed!) |

---

## ✅ Task 3 — Music Generation with AI
| Feature | Detail |
|---------|--------|
| Generator | Markov Chain (instant) + LSTM (trainable) |
| Scales | C Major, A Minor, Blues, Pentatonic, Dorian... |
| Export | MIDI (.mid) and MusicXML (.xml) |
| Run | `pip install music21 tensorflow numpy` → `python music_gen.py` |

---

## ✅ Task 4 — Object Detection and Tracking
| Feature | Detail |
|---------|--------|
| Model | YOLOv8n (auto-downloads, ~6MB) |
| Tracker | Custom Centroid Tracker (SORT-like) |
| Input | Webcam or video file |
| Run | `pip install ultralytics opencv-python Pillow numpy` → `python object_detection.py` |

---

## 🚀 Quick Setup (All Tasks)

```bash
# Clone and install all dependencies
pip install deep-translator pyttsx3 music21 tensorflow numpy ultralytics opencv-python Pillow

# Run any task
python Task1_LanguageTranslation/app.py
python Task2_ChatbotFAQ/chatbot.py
python Task3_MusicGeneration/music_gen.py
python Task4_ObjectDetection/object_detection.py
```

---

## 📞 CodeAlpha Contact
- 🌐 Website: [www.codealpha.tech](http://www.codealpha.tech)
- 📧 Email: services@codealpha.tech
- 📱 WhatsApp: +91 9336576683
