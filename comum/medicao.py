"""Medição real de tempo (time.perf_counter) e memória de pico (tracemalloc)."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Medicao:
    resultado: Any
    tempo_s: float
    memoria_pico_kb: float


def medir(func: Callable, *args, **kwargs) -> Medicao:
    """Executa func(*args, **kwargs) medindo tempo de parede e pico de memória.

    Observação: tracemalloc rastreia as alocações feitas pelo Python e pelo
    NumPy (que registra seus buffers no tracemalloc), mas adiciona overhead;
    por isso o tempo é medido numa execução separada, sem tracemalloc ativo.
    """
    inicio = time.perf_counter()
    resultado = func(*args, **kwargs)
    tempo = time.perf_counter() - inicio

    tracemalloc.start()
    func(*args, **kwargs)
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Medicao(resultado, tempo, pico / 1024)
