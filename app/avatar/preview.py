"""Optional 1x tk preview. Not the product UI — a stand-in for the corner widget."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image, ImageTk


class TkFrameSink:
    def __init__(self, title: str = "Kafka") -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)
        self.root.configure(bg="#1b1b1f")
        stage = tk.Frame(self.root, bg="#ffffff", padx=2, pady=2)
        stage.pack(padx=8, pady=8)
        self.view = tk.Label(stage, bg="#ffffff", bd=0)
        self.view.pack()
        self._photo: ImageTk.PhotoImage | None = None
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    @property
    def closed(self) -> bool:
        return self._closed

    def push_frame(self, rgb: np.ndarray) -> None:
        if self._closed:
            return
        image = Image.fromarray(rgb, mode="RGB")
        self._photo = ImageTk.PhotoImage(image)
        self.view.configure(image=self._photo)
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass
