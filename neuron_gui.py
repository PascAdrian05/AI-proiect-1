"""Interfața grafică (tkinter + ttk) pentru simulatorul de neuron."""
import tkinter as tk
from tkinter import ttk

from neuron_math import (ACTIVATIONS, GIN_METHODS, THETA_ONLY, activate,
                         compute_gin, parse_float, to_binary)

MAX_INPUTS = 20


class NeuronApp(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, padding=10)
        self.pack(fill="both", expand=True)
        self.x_vars, self.w_vars = [], []  # variabilele rândurilor curente

        self._create_variables()
        self._build_inputs_section()
        self._build_gin_section()
        self._build_activation_section()
        self._build_result_section()

        self.rebuild_rows()
        self.update_g_state()
        self.update()

    # ----- variabile ------------------------------------------------------
    def _create_variables(self):
        self.n_var = tk.StringVar(value="3")
        self.gin_method = tk.StringVar(value="Sum")
        self.theta_var = tk.StringVar(value="0.0")
        self.g_var = tk.StringVar(value="1.0")
        self.act_var = tk.StringVar(value="Linear")
        self.out_type = tk.StringVar(value="Real")
        self.gin_text = tk.StringVar()
        self.out_text = tk.StringVar()

        # trace_add => recalculare automată la orice modificare
        for v in (self.gin_method, self.theta_var, self.g_var,
                  self.act_var, self.out_type):
            v.trace_add("write", self.update)
        self.act_var.trace_add("write", self.update_g_state)
        self.n_var.trace_add("write", self.on_n_change)

    # ----- secțiuni GUI ---------------------------------------------------
    def _build_inputs_section(self):
        frame = ttk.LabelFrame(self, text="1. Configurare Intrări", padding=8)
        frame.pack(fill="both", expand=True)

        top = ttk.Frame(frame)
        top.pack(fill="x")
        ttk.Label(top, text="Număr intrări (n):").pack(side="left")
        ttk.Spinbox(top, from_=1, to=MAX_INPUTS, width=5,
                    textvariable=self.n_var,
                    command=self.on_n_change).pack(side="left", padx=5)

        # Canvas + Scrollbar pentru rândurile de intrări
        holder = ttk.Frame(frame)
        holder.pack(fill="both", expand=True, pady=5)
        canvas = tk.Canvas(holder, height=160, highlightthickness=0)
        scroll = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.rows_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        self.rows_frame.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

    def _build_gin_section(self):
        frame = ttk.LabelFrame(self, text="2. Global Input", padding=8)
        frame.pack(fill="x", pady=5)
        ttk.Label(frame, text="Metodă GIN:").pack(side="left")
        ttk.Combobox(frame, values=GIN_METHODS, textvariable=self.gin_method,
                     state="readonly", width=8).pack(side="left", padx=5)
        ttk.Label(frame, text="GIN =").pack(side="left", padx=(15, 2))
        ttk.Label(frame, textvariable=self.gin_text,
                  font=("Segoe UI", 12, "bold")).pack(side="left")

    def _build_activation_section(self):
        frame = ttk.LabelFrame(self, text="3. Activare și Parametri", padding=8)
        frame.pack(fill="x", pady=5)
        ttk.Label(frame, text="Funcție:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(frame, values=ACTIVATIONS, textvariable=self.act_var,
                     state="readonly", width=10).grid(row=0, column=1, padx=5)
        ttk.Label(frame, text="Prag θ:").grid(row=0, column=2, padx=(15, 2))
        ttk.Entry(frame, textvariable=self.theta_var,
                  width=8).grid(row=0, column=3)
        ttk.Label(frame, text="Factor g:").grid(row=0, column=4, padx=(15, 2))
        self.g_entry = ttk.Entry(frame, textvariable=self.g_var, width=8)
        self.g_entry.grid(row=0, column=5)

    def _build_result_section(self):
        frame = ttk.LabelFrame(self, text="4. Rezultat Final", padding=8)
        frame.pack(fill="x", pady=5)
        ttk.Label(frame, text="Tip ieșire:").pack(side="left")
        for t in ("Real", "Binar"):
            ttk.Radiobutton(frame, text=t, value=t,
                            variable=self.out_type).pack(side="left", padx=5)
        ttk.Label(frame, text="Output:").pack(side="left", padx=(15, 2))
        ttk.Label(frame, textvariable=self.out_text, foreground="#0a5",
                  font=("Segoe UI", 22, "bold")).pack(side="left")

    # ----- logică GUI -----------------------------------------------------
    def on_n_change(self, *_):
        """Regenerează rândurile când se schimbă n (valid doar 1..20)."""
        try:
            n = int(self.n_var.get())
        except ValueError:
            return  # câmp gol / în curs de scriere
        if 1 <= n <= MAX_INPUTS and n != len(self.x_vars):
            self.rebuild_rows()
            self.update()

    def rebuild_rows(self):
        """Creează n rânduri, păstrând valorile deja introduse."""
        n = int(self.n_var.get())
        old_x = [v.get() for v in self.x_vars]
        old_w = [v.get() for v in self.w_vars]
        for child in self.rows_frame.winfo_children():
            child.destroy()
        self.x_vars, self.w_vars = [], []

        ttk.Label(self.rows_frame, text="x_i").grid(row=0, column=1)
        ttk.Label(self.rows_frame, text="w_i").grid(row=0, column=2)
        for i in range(n):
            xv = tk.StringVar(value=old_x[i] if i < len(old_x) else "1.0")
            wv = tk.StringVar(value=old_w[i] if i < len(old_w) else "0.5")
            xv.trace_add("write", self.update)
            wv.trace_add("write", self.update)
            ttk.Label(self.rows_frame, text=f"Intrarea {i + 1}:").grid(
                row=i + 1, column=0, sticky="w", padx=5)
            ttk.Entry(self.rows_frame, textvariable=xv, width=10).grid(
                row=i + 1, column=1, padx=5, pady=1)
            ttk.Entry(self.rows_frame, textvariable=wv, width=10).grid(
                row=i + 1, column=2, padx=5, pady=1)
            self.x_vars.append(xv)
            self.w_vars.append(wv)

    def update_g_state(self, *_):
        """Step și Signum nu folosesc g => Entry dezactivat."""
        disabled = self.act_var.get() in THETA_ONLY
        self.g_entry.configure(state="disabled" if disabled else "normal")

    def update(self, *_):
        """Recalculează GIN și Output la orice modificare."""
        if not self.x_vars:
            return
        xs = [parse_float(v.get()) for v in self.x_vars]
        ws = [parse_float(v.get()) for v in self.w_vars]
        func = self.act_var.get()
        try:
            gin = compute_gin(xs, ws, self.gin_method.get())
            y = activate(gin, parse_float(self.theta_var.get()),
                         parse_float(self.g_var.get()), func)
            self.gin_text.set(f"{gin:.6g}")
            if self.out_type.get() == "Binar":
                self.out_text.set(str(to_binary(y, func)))
            else:
                self.out_text.set(f"{y:.6f}")
        except (ValueError, OverflowError):
            self.gin_text.set("eroare")
            self.out_text.set("eroare")
