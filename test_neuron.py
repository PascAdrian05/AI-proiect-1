"""Teste pentru neuron_math și neuron_gui. Rulare: python -m unittest -v test_neuron"""
import math
import tkinter as tk
import unittest

from neuron_gui import NeuronApp
from neuron_math import activate, compute_gin, parse_float, to_binary


class TestMath(unittest.TestCase):
    def test_parse_float(self):
        self.assertEqual(parse_float("2.5"), 2.5)
        self.assertEqual(parse_float("2,5"), 2.5)
        self.assertEqual(parse_float(""), 0.0)
        self.assertEqual(parse_float("-"), 0.0)
        self.assertEqual(parse_float("abc"), 0.0)

    def test_gin(self):
        xs, ws = [1, 2, 3], [0.5, 1, -1]  # p = 0.5, 2, -3
        self.assertAlmostEqual(compute_gin(xs, ws, "Sum"), -0.5)
        self.assertAlmostEqual(compute_gin(xs, ws, "Prod"), -3.0)
        self.assertAlmostEqual(compute_gin(xs, ws, "Min"), -3.0)
        self.assertAlmostEqual(compute_gin(xs, ws, "Max"), 2.0)

    def test_step_signum(self):
        self.assertEqual(activate(1, 1, 9, "Step"), 1.0)
        self.assertEqual(activate(0.9, 1, 9, "Step"), 0.0)
        self.assertEqual(activate(2, 1, 9, "Signum"), 1.0)
        self.assertEqual(activate(1, 1, 9, "Signum"), 0.0)
        self.assertEqual(activate(0, 1, 9, "Signum"), -1.0)

    def test_linear_sigmoid_tanh(self):
        self.assertAlmostEqual(activate(3, 1, 2, "Linear"), 4.0)
        self.assertAlmostEqual(activate(1, 1, 2, "Sigmoid"), 0.5)
        self.assertAlmostEqual(activate(3, 1, 2, "Tanh"), math.tanh(4))

    def test_overflow_safe(self):
        self.assertAlmostEqual(activate(-1e9, 0, 1, "Sigmoid"), 0.0)
        self.assertAlmostEqual(activate(1e9, 0, 1, "Sigmoid"), 1.0)
        self.assertEqual(activate(1e9, 0, 1, "Tanh"), 1.0)

    def test_binary(self):
        self.assertEqual(to_binary(0.5, "Sigmoid"), 1)
        self.assertEqual(to_binary(0.49, "Sigmoid"), 0)
        self.assertEqual(to_binary(0.0, "Step"), 0)
        self.assertEqual(to_binary(0.0, "Tanh"), 1)
        self.assertEqual(to_binary(-0.1, "Tanh"), -1)
        self.assertEqual(to_binary(-1.0, "Signum"), -1)
        self.assertEqual(to_binary(0.0, "Signum"), 1)
        self.assertEqual(to_binary(0.0, "Linear"), 1)
        self.assertEqual(to_binary(-2.0, "Linear"), 0)


class TestGui(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = NeuronApp(self.root)

    def tearDown(self):
        self.root.destroy()

    def test_defaults(self):
        # 3 intrări, x=1, w=0.5 => GIN=1.5; Linear g=1 theta=0 => 1.5
        self.assertEqual(len(self.app.x_vars), 3)
        self.assertEqual(self.app.gin_text.get(), "1.5")
        self.assertEqual(self.app.out_text.get(), "1.500000")

    def test_live_update_input(self):
        self.app.x_vars[0].set("3")
        self.assertEqual(self.app.gin_text.get(), "2.5")

    def test_empty_and_minus(self):
        self.app.x_vars[0].set("")
        self.assertEqual(self.app.gin_text.get(), "1")
        self.app.w_vars[1].set("-")
        self.assertEqual(self.app.gin_text.get(), "0.5")

    def test_change_n_keeps_values(self):
        self.app.x_vars[0].set("4")
        self.app.n_var.set("5")
        self.assertEqual(len(self.app.x_vars), 5)
        self.assertEqual(self.app.x_vars[0].get(), "4")
        self.app.n_var.set("2")
        self.assertEqual(len(self.app.w_vars), 2)
        self.app.n_var.set("")      # gol: nu crapă
        self.app.n_var.set("99")    # peste maxim: ignorat
        self.assertEqual(len(self.app.x_vars), 2)

    def test_method_and_activation(self):
        self.app.gin_method.set("Prod")
        self.assertEqual(self.app.gin_text.get(), "0.125")
        self.app.act_var.set("Step")
        self.assertEqual(str(self.app.g_entry.cget("state")), "disabled")
        self.assertEqual(self.app.out_text.get(), "1.000000")
        self.app.act_var.set("Sigmoid")
        self.assertEqual(str(self.app.g_entry.cget("state")), "normal")

    def test_binary_output(self):
        self.app.act_var.set("Sigmoid")
        self.app.out_type.set("Binar")
        self.assertEqual(self.app.out_text.get(), "1")
        self.app.theta_var.set("10")
        self.assertEqual(self.app.out_text.get(), "0")

    def test_bad_theta_text(self):
        self.app.theta_var.set("-")
        self.assertEqual(self.app.out_text.get(), "1.500000")


if __name__ == "__main__":
    unittest.main()
