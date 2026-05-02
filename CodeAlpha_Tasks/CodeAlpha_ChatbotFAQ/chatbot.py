"""
CodeAlpha Internship - Task 2: Chatbot for FAQs
Uses NLTK + TF-IDF cosine similarity for FAQ matching
"""

import tkinter as tk
from tkinter import scrolledtext
import threading
import json
import re
import math
from collections import Counter

# ── Sample FAQ Data ──────────────────────────────────────────────────────────
FAQ_DATA = [
    {"q": "What is CodeAlpha?",
     "a": "CodeAlpha is a leading software development company dedicated to driving innovation and excellence across emerging technologies."},
    {"q": "How do I apply for the internship?",
     "a": "You can apply for the internship through the CodeAlpha website at www.codealpha.tech or via the WhatsApp group."},
    {"q": "What is the duration of the internship?",
     "a": "The internship duration is typically 1 month, with tasks assigned at the start."},
    {"q": "What tasks are assigned in the AI internship?",
     "a": "AI interns are assigned 4 tasks: Language Translation Tool, Chatbot for FAQs, Music Generation with AI, and Object Detection & Tracking. You must complete at least 2 or 3."},
    {"q": "Will I get a certificate?",
     "a": "Yes! Upon completing the required tasks, you will receive a QR-verified Completion Certificate, Offer Letter, and optionally a Letter of Recommendation."},
    {"q": "How do I submit my project?",
     "a": "Upload your source code to GitHub (repo name: CodeAlpha_ProjectName), post a video on LinkedIn tagging @CodeAlpha, and submit via the official Submission Form shared in your WhatsApp group."},
    {"q": "Is the internship paid?",
     "a": "The internship may be unpaid but provides valuable perks including certificates, LOR, placement support, and resume building assistance."},
    {"q": "What programming languages can I use?",
     "a": "Interns typically use Python for AI tasks. Libraries like NLTK, SpaCy, OpenCV, TensorFlow, and PyTorch are commonly used."},
    {"q": "How do I contact CodeAlpha?",
     "a": "Email: services@codealpha.tech | WhatsApp: +91 9336576683 | Website: www.codealpha.tech"},
    {"q": "What happens if I submit only one task?",
     "a": "Submitting only one task is considered incomplete. You must complete at least 2 or 3 tasks to be eligible for the certificate."},
    {"q": "Can I work on all 4 tasks?",
     "a": "Yes! While only 2-3 tasks are required, you are welcome to complete all 4 for more experience and a stronger portfolio."},
    {"q": "What is object detection?",
     "a": "Object detection is a computer vision technique that identifies and locates objects within images or video. In Task 4, you'll use YOLO or Faster R-CNN with OpenCV."},
    {"q": "How do I generate music with AI?",
     "a": "In Task 3, you collect MIDI data, preprocess it with music21, then train an LSTM or GAN model to generate new music sequences."},
    {"q": "What is cosine similarity?",
     "a": "Cosine similarity measures the angle between two vectors in a multi-dimensional space. In NLP, it's used to find how similar two text documents are based on word frequencies."},
    {"q": "Hello",
     "a": "Hello! 👋 I'm the CodeAlpha FAQ Bot. Ask me anything about the internship program!"},
    {"q": "Who are you?",
     "a": "I'm an AI-powered FAQ chatbot built for the CodeAlpha AI Internship Task 2. I can answer questions about the internship, tasks, and submission process."},
    {"q": "Thank you",
     "a": "You're welcome! 😊 Best of luck with your CodeAlpha internship tasks!"},
]

# ── Simple NLP Pipeline (no external dependencies) ───────────────────────────
STOPWORDS = {
    "i","me","my","we","our","you","your","he","she","it","they","them","their",
    "what","which","who","is","are","was","were","be","been","being","have","has",
    "do","does","did","will","would","could","should","may","might","a","an","the",
    "and","or","but","in","on","at","to","for","of","with","by","from","about",
    "how","why","when","where","this","that","these","those","can","get","if","not"
}

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def tfidf_vectors(corpus_tokens, query_tokens):
    """Compute TF-IDF cosine similarity between query and each document."""
    all_docs = corpus_tokens + [query_tokens]
    N = len(all_docs)
    # IDF
    df = Counter()
    for doc in all_docs:
        for term in set(doc):
            df[term] += 1
    idf = {term: math.log(N / (1 + count)) for term, count in df.items()}
    # TF-IDF
    def tfidf(doc):
        tf = Counter(doc)
        total = max(len(doc), 1)
        return {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}
    return [tfidf(doc) for doc in all_docs]

def cosine_sim(v1, v2):
    keys = set(v1) | set(v2)
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
    mag1 = math.sqrt(sum(x**2 for x in v1.values()))
    mag2 = math.sqrt(sum(x**2 for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

class FAQChatbot:
    def __init__(self):
        self.questions = [item["q"] for item in FAQ_DATA]
        self.answers = [item["a"] for item in FAQ_DATA]
        self.corpus_tokens = [tokenize(q) for q in self.questions]

    def get_response(self, user_input):
        query_tokens = tokenize(user_input)
        if not query_tokens:
            return "I didn't understand that. Could you rephrase?"
        vectors = tfidf_vectors(self.corpus_tokens, query_tokens)
        query_vec = vectors[-1]
        doc_vecs = vectors[:-1]
        scores = [cosine_sim(doc_vec, query_vec) for doc_vec in doc_vecs]
        best_idx = scores.index(max(scores))
        best_score = max(scores)
        if best_score < 0.05:
            return ("I'm not sure about that. Try asking about:\n"
                    "• Internship tasks  • Certificate  • Submission\n"
                    "• Contact info  • Duration  • How to apply")
        return self.answers[best_idx]


# ── GUI ───────────────────────────────────────────────────────────────────────
class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 CodeAlpha FAQ Chatbot")
        self.root.geometry("700x580")
        self.root.configure(bg="#0d1117")
        self.bot = FAQChatbot()
        self._build_ui()
        self._bot_message("Hello! 👋 I'm the CodeAlpha FAQ Bot.\nAsk me anything about the internship program!")

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#161b22", pady=12)
        header.pack(fill="x")
        tk.Label(header, text="🤖 CodeAlpha FAQ Chatbot",
                 font=("Courier New", 16, "bold"), fg="#58a6ff", bg="#161b22").pack()
        tk.Label(header, text="CodeAlpha AI Internship — Task 2 | NLP + TF-IDF Cosine Similarity",
                 font=("Courier New", 8), fg="#8b949e", bg="#161b22").pack()

        # Chat area
        chat_frame = tk.Frame(self.root, bg="#0d1117")
        chat_frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.chat_area = scrolledtext.ScrolledText(
            chat_frame, wrap="word", font=("Courier New", 11),
            bg="#161b22", fg="#c9d1d9", insertbackground="white",
            relief="flat", padx=12, pady=12, state="disabled",
            cursor="arrow"
        )
        self.chat_area.pack(fill="both", expand=True)
        self.chat_area.tag_config("user", foreground="#58a6ff", font=("Courier New", 11, "bold"))
        self.chat_area.tag_config("bot", foreground="#3fb950", font=("Courier New", 11))
        self.chat_area.tag_config("label_user", foreground="#388bfd", font=("Courier New", 9, "bold"))
        self.chat_area.tag_config("label_bot", foreground="#2ea043", font=("Courier New", 9, "bold"))
        self.chat_area.tag_config("separator", foreground="#30363d")

        # FAQ Suggestions
        sug_frame = tk.Frame(self.root, bg="#0d1117")
        sug_frame.pack(fill="x", padx=15)
        tk.Label(sug_frame, text="💡 Quick Questions:", font=("Courier New", 9),
                 fg="#8b949e", bg="#0d1117").pack(side="left")
        suggestions = ["Certificate?", "How to submit?", "Contact?", "Tasks list?"]
        for s in suggestions:
            btn = tk.Button(sug_frame, text=s, font=("Courier New", 8),
                            bg="#21262d", fg="#58a6ff", relief="flat",
                            padx=8, pady=2, cursor="hand2",
                            command=lambda q=s: self._send_message(q))
            btn.pack(side="left", padx=3, pady=4)

        # Input area
        input_frame = tk.Frame(self.root, bg="#161b22", pady=10)
        input_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(
            input_frame, textvariable=self.input_var,
            font=("Courier New", 12), bg="#21262d", fg="#c9d1d9",
            insertbackground="white", relief="flat", bd=0
        )
        self.input_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(10, 5))
        self.input_entry.bind("<Return>", lambda e: self._send_message())

        send_btn = tk.Button(
            input_frame, text="Send ▶", font=("Courier New", 11, "bold"),
            bg="#238636", fg="white", relief="flat", padx=15, pady=6,
            cursor="hand2", command=self._send_message
        )
        send_btn.pack(side="right", padx=(5, 10))

    def _send_message(self, text=None):
        msg = text or self.input_var.get().strip()
        if not msg:
            return
        self.input_var.set("")
        self._user_message(msg)
        threading.Thread(target=self._respond, args=(msg,), daemon=True).start()

    def _respond(self, msg):
        self.root.after(400, lambda: self._bot_message(self.bot.get_response(msg)))

    def _user_message(self, msg):
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", "\nYou  ", "label_user")
        self.chat_area.insert("end", f"{msg}\n", "user")
        self.chat_area.config(state="disabled")
        self.chat_area.see("end")

    def _bot_message(self, msg):
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", "\nBot  ", "label_bot")
        self.chat_area.insert("end", f"{msg}\n", "bot")
        self.chat_area.insert("end", "─" * 60 + "\n", "separator")
        self.chat_area.config(state="disabled")
        self.chat_area.see("end")


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
