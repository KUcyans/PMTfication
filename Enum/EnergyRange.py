from enum import Enum
from .Flavour import Flavour

class EnergyRange(Enum):
    ER_100_GEV_10_TEV = (r"$100\,\text{GeV} - 10\,\text{TeV}$", 
                         "100GeV-10TeV",  # Pure string version for easy reference
                         {Flavour.E: "22013", Flavour.MU: "22010", Flavour.TAU: "22016"})
    ER_10_TEV_1_PEV = (r"$10\,\text{TeV} - 1\,\text{PeV}$", 
                       "10TeV-1PeV",
                       {Flavour.E: "22014", Flavour.MU: "22011", Flavour.TAU: "22017"})
    ER_1_PEV_100_PEV = (r"$1\,\text{PeV} - 100\,\text{PeV}$", 
                        "1PeV-100PeV",
                        {Flavour.E: "22015", Flavour.MU: "22012", Flavour.TAU: "22018"})

    def __init__(self, latex: str, string: str, subdirs: dict):
        self._latex = latex
        self._string = string
        self._subdirs = subdirs

    @property
    def latex(self) -> str:
        """Return the LaTeX expression for the energy range."""
        return self._latex
    
    @property
    def string(self) -> str:
        """Return the simple string version for the energy range."""
        return self._string

    def __getattr__(self, name: str):
        """Allow access via .E, .MU, .TAU for subdirectories."""
        if name not in Flavour.__members__:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        flavour = getattr(Flavour, name)
        return self._subdirs.get(flavour, None)
    
    @staticmethod
    def get_energy_range(subdir: str) -> "EnergyRange":
        """Get the corresponding EnergyRange by subdir value."""
        for energy_range in EnergyRange:
            for flavour, sub in energy_range._subdirs.items():
                if sub == subdir:
                    return energy_range
        return None

    @staticmethod
    def get_flavour(subdir: str) -> "Flavour":
        """Get the corresponding Flavour by subdir value."""
        for energy_range in EnergyRange:
            for flavour, sub in energy_range._subdirs.items():
                if sub == subdir:
                    return flavour
        return None

    @staticmethod
    def get_subdir(energy_range: "EnergyRange", flavour: "Flavour") -> str:
        """Get the subdirectory string for a given EnergyRange and Flavour."""
        return energy_range._subdirs.get(flavour, None)
