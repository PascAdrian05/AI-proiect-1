"""Logica matematică a neuronului (fără dependență de GUI)."""
import math

GIN_METHODS = ["Sum", "Prod", "Min", "Max"]
ACTIVATIONS = ["Linear", "Sigmoid", "Signum", "Tanh", "Step"]
# Funcțiile care folosesc doar theta (g este ignorat)
THETA_ONLY = ("Step", "Signum")


def parse_float(text):
    """Convertește text în float; gol sau '-' (în curs de scriere) => 0.0."""
    try:
        return float(text.strip().replace(",", "."))
    except ValueError:
        return 0.0


def compute_gin(xs, ws, method):
    """Calculează Global Input din produsele p_i = x_i * w_i."""
    products = [x * w for x, w in zip(xs, ws)]
    if not products:
        return 0.0
    if method == "Sum":
        return sum(products)
    if method == "Prod":
        return math.prod(products)
    if method == "Min":
        return min(products)
    return max(products)


def activate(x, theta, g, func):
    """Aplică funcția de activare asupra lui x (GIN)."""
    d = x - theta
    if func == "Step":
        return 1.0 if d >= 0 else 0.0
    if func == "Signum":
        if d > 0:
            return 1.0
        elif d < 0:
            return -1.0
        else:
            return 0.0
    z =g*d
    if z>500:
        z = 500.0 
    if z<-500:
        z = -500.0
             # limităm z pentru a evita OverflowError
    if func == "Linear":
        return g * d
    if func == "Sigmoid":
        return 1.0 / (1.0 + math.exp(-z))
    return math.tanh(z)  # Tanh


def to_binary(y, func):
    """Transformă ieșirea y într-o valoare binară (0 sau 1) sau semn (-1 sau 1)."""
    if func == "Sigmoid" or func == "Step":
        if y >= 0.5:
            return 1
        else:
            return 0

   
    elif func == "Tanh" or func == "Signum":
        if y >= 0:
            return 1
        else:
            return -1

    elif func == "Linear":
        if y >= 0:
            return 1
        else:
            return 0