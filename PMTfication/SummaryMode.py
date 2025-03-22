from enum import Enum

class SummaryMode(Enum):
    CLASSIC = (0, 'thorsten', 5)
    SECOND = (1, 'second', 5)
    EQUINOX = (2, 'third', 5)
    
    def __init__(self, index: int, name: str, n_first_pulse_collect: int):
        self._index = index
        self._name = name
        self._n_first_pulse_collect = n_first_pulse_collect
    
    @property
    def index(self) -> int:
        """Return the index of the summary mode."""
        return self._index
    # example usage: print(SummaryMode.CLASSIC.index)
    
    
    def __str__(self) -> str:
        return self._name
        
    @property
    def n_collect(self) -> int:
        """Return the number of first pulses to collect."""
        return self._n_first_pulse_collect
    
    
    
    @staticmethod
    def from_index(index: int) -> 'SummaryMode':
        """Return the summary mode from the index."""
        for mode in SummaryMode:
            if mode.index == index:
                return mode
        raise ValueError(f"Invalid index: {index}")