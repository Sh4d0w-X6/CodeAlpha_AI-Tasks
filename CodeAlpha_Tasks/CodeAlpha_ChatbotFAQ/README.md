# 🤖 Task 2 — Chatbot for FAQs
**CodeAlpha AI Internship**

## Features
- 15+ FAQs about the CodeAlpha internship program
- Custom NLP pipeline (tokenization, stopword removal)
- **TF-IDF vectorization** + **Cosine Similarity** matching — no external ML library needed
- Quick-suggestion buttons for common questions
- Dark-themed terminal-style chat UI

## Setup & Run
```bash
python chatbot.py
```
> No extra dependencies! Pure Python standard library + Tkinter.

## How It Works
1. User types a question
2. Text is tokenized and cleaned (stopwords removed)
3. TF-IDF vectors computed for all FAQ questions + user query
4. Cosine similarity score calculated for each FAQ
5. Best-matching FAQ answer returned

## Tech Stack
- **Python** (standard library only)
- **Tkinter** (GUI)
- Custom TF-IDF + Cosine Similarity (no scikit-learn needed)

## Extending the FAQ
Edit the `FAQ_DATA` list in `chatbot.py` to add more Q&A pairs.
