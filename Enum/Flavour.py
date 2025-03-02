from enum import Enum

class Flavour(Enum):
    E = ("nu_e", r"\nu_e", 12)
    MU = ("nu_mu", r"\nu_\mu", 14)
    TAU = ("nu_tau", r"\nu_\tau", 16)

    def __init__(self, alias: str, latex: str, pdg: int):
        self.alias = alias
        self.latex = latex
        self.pdg = pdg