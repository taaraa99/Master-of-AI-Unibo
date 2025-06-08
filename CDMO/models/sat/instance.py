#Functions and classes for reading the instances
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass(slots=True)
class Instance:
    m: int
    n: int
    cap: List[int]
    size: List[int]
    D: List[List[int]]

    @property
    def depot(self) -> int:
        return self.n


def load_instance(path: Path) -> Instance:
    tok = path.read_text().split()
    it = iter(tok)
    m = int(next(it)); n = int(next(it))
    cap = [int(next(it)) for _ in range(m)]
    size = [int(next(it)) for _ in range(n)]
    flat = [int(next(it)) for _ in range((n+1)*(n+1))]
    D = [flat[r*(n+1):(r+1)*(n+1)] for r in range(n+1)]
    return Instance(m, n, cap, size, D)