"""
CodeAlpha Internship - Task 1: Language Translation Tool
Uses deep-translator (free, no API key required) + Tkinter GUI
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading

try:
    from deep_translator import GoogleTranslator
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deep-translator"])
    from deep_translator import GoogleTranslator

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Language map
LANGUAGES = {
    "Auto Detect": "auto",
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh-CN",
    "Arabic": "ar",
    "Japanese": "ja",
    "Korean": "ko",
    "Portuguese": "pt",
    "Russian": "ru",
    "Italian": "it",
    "Dutch": "nl",
    "Turkish": "tr",
    "Bengali": "bn",
    "Urdu": "ur",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
}


class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌐 CodeAlpha Language Translator")
        self.root.geometry("800x600")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(True, True)

        self._build_ui()

    def _build_ui(self):
        # Title
        title = tk.Label(
            self.root,
            text="🌐 Language Translation Tool",
            font=("Helvetica", 20, "bold"),
            fg="#e94560",
            bg="#1a1a2e",
        )
        title.pack(pady=(20, 5))

        subtitle = tk.Label(
            self.root,
            text="CodeAlpha AI Internship — Task 1",
            font=("Helvetica", 10),
            fg="#888",
            bg="#1a1a2e",
        )
        subtitle.pack(pady=(0, 15))

        # Language selection frame
        lang_frame = tk.Frame(self.root, bg="#1a1a2e")
        lang_frame.pack(fill="x", padx=30, pady=5)

        tk.Label(lang_frame, text="From:", font=("Helvetica", 11, "bold"),
                 fg="#e0e0e0", bg="#1a1a2e").grid(row=0, column=0, padx=5)
        self.src_lang = ttk.Combobox(lang_frame, values=list(LANGUAGES.keys()),
                                     width=20, state="readonly", font=("Helvetica", 10))
        self.src_lang.set("Auto Detect")
        self.src_lang.grid(row=0, column=1, padx=10)

        tk.Label(lang_frame, text="→", font=("Helvetica", 16, "bold"),
                 fg="#e94560", bg="#1a1a2e").grid(row=0, column=2, padx=10)

        tk.Label(lang_frame, text="To:", font=("Helvetica", 11, "bold"),
                 fg="#e0e0e0", bg="#1a1a2e").grid(row=0, column=3, padx=5)
        self.tgt_lang = ttk.Combobox(lang_frame, values=list(LANGUAGES.keys())[1:],
                                     width=20, state="readonly", font=("Helvetica", 10))
        self.tgt_lang.set("Hindi")
        self.tgt_lang.grid(row=0, column=4, padx=10)

        # Swap button
        swap_btn = tk.Button(lang_frame, text="⇄ Swap", command=self._swap_langs,
                             bg="#e94560", fg="white", font=("Helvetica", 9, "bold"),
                             relief="flat", padx=8, pady=3, cursor="hand2")
        swap_btn.grid(row=0, column=5, padx=10)

        # Text areas
        text_frame = tk.Frame(self.root, bg="#1a1a2e")
        text_frame.pack(fill="both", expand=True, padx=30, pady=10)

        # Input
        tk.Label(text_frame, text="Input Text", font=("Helvetica", 10, "bold"),
                 fg="#aaa", bg="#1a1a2e").grid(row=0, column=0, sticky="w", padx=5)
        self.input_text = tk.Text(text_frame, height=10, wrap="word",
                                  font=("Helvetica", 12), bg="#16213e", fg="#e0e0e0",
                                  insertbackground="white", relief="flat",
                                  padx=10, pady=10, bd=2)
        self.input_text.grid(row=1, column=0, padx=5, sticky="nsew")

        # Output
        tk.Label(text_frame, text="Translated Text", font=("Helvetica", 10, "bold"),
                 fg="#aaa", bg="#1a1a2e").grid(row=0, column=1, sticky="w", padx=5)
        self.output_text = tk.Text(text_frame, height=10, wrap="word",
                                   font=("Helvetica", 12), bg="#16213e", fg="#4ecca3",
                                   relief="flat", padx=10, pady=10, bd=2, state="disabled")
        self.output_text.grid(row=1, column=1, padx=5, sticky="nsew")

        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid_columnconfigure(1, weight=1)
        text_frame.grid_rowconfigure(1, weight=1)

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#1a1a2e")
        btn_frame.pack(pady=10)

        translate_btn = tk.Button(btn_frame, text="🔄 Translate", command=self._translate,
                                  bg="#e94560", fg="white", font=("Helvetica", 12, "bold"),
                                  relief="flat", padx=20, pady=8, cursor="hand2")
        translate_btn.grid(row=0, column=0, padx=10)

        copy_btn = tk.Button(btn_frame, text="📋 Copy", command=self._copy,
                             bg="#4ecca3", fg="#1a1a2e", font=("Helvetica", 11, "bold"),
                             relief="flat", padx=15, pady=8, cursor="hand2")
        copy_btn.grid(row=0, column=1, padx=10)

        clear_btn = tk.Button(btn_frame, text="🗑 Clear", command=self._clear,
                              bg="#555", fg="white", font=("Helvetica", 11),
                              relief="flat", padx=15, pady=8, cursor="hand2")
        clear_btn.grid(row=0, column=2, padx=10)

        if TTS_AVAILABLE:
            tts_btn = tk.Button(btn_frame, text="🔊 Speak", command=self._speak,
                                bg="#f0a500", fg="#1a1a2e", font=("Helvetica", 11, "bold"),
                                relief="flat", padx=15, pady=8, cursor="hand2")
            tts_btn.grid(row=0, column=3, padx=10)

        # Status bar
        self.status = tk.Label(self.root, text="Ready", font=("Helvetica", 9),
                               fg="#888", bg="#0f3460", anchor="w", padx=10)
        self.status.pack(fill="x", side="bottom")

    def _swap_langs(self):
        src = self.src_lang.get()
        tgt = self.tgt_lang.get()
        if src != "Auto Detect":
            self.src_lang.set(tgt)
            self.tgt_lang.set(src)

    def _translate(self):
        text = self.input_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Empty Input", "Please enter some text to translate.")
            return
        self.status.config(text="⏳ Translating...")
        self.root.update()
        threading.Thread(target=self._do_translate, args=(text,), daemon=True).start()

    def _do_translate(self, text):
        try:
            src = LANGUAGES[self.src_lang.get()]
            tgt = LANGUAGES[self.tgt_lang.get()]
            result = GoogleTranslator(source=src, target=tgt).translate(text)
            self.output_text.config(state="normal")
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", result)
            self.output_text.config(state="disabled")
            self.status.config(text=f"✅ Translated to {self.tgt_lang.get()}")
        except Exception as e:
            self.status.config(text=f"❌ Error: {str(e)}")
            messagebox.showerror("Translation Error", str(e))

    def _copy(self):
        text = self.output_text.get("1.0", "end").strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status.config(text="📋 Copied to clipboard!")

    def _clear(self):
        self.input_text.delete("1.0", "end")
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.config(state="disabled")
        self.status.config(text="Cleared")

    def _speak(self):
        text = self.output_text.get("1.0", "end").strip()
        if text and TTS_AVAILABLE:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()


if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
