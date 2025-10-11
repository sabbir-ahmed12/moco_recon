from abc import ABC, abstractmethod

class Recon(ABC):
    """Interface class for the reconstruction algorithms."""

    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        """Execute the reconstruction and return outputs."""
        pass
