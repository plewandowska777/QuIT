import numpy as np


class ComplexSpace:
    def __init__(self, dimension: int, name: str) -> None:
        self.dimension = dimension
        self.name = name

        def ket(self, position: int):
            vector = np.zeros(self.dimension)
            vector[position] = 1
            return vector
