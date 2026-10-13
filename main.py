"""Punct de intrare: rulează cu `python main.py`."""
import tkinter as tk

from neuron_gui import NeuronApp

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Neuron Artificial")
    NeuronApp(root)
    root.mainloop()
