"""Etapa A — Brute Force: ótimo EXATO num subconjunto pequeno (benchmark do GA).

Enumera TODAS as combinações de níveis permitidos: N = prod_i (L - nivel_min_i) <= L^n.
Com L = 7 e n = 6, N <= 117.649; com n = 8, N <= 5,7 milhões; com n = 30 seriam
7^30 ~ 2,25e25 combinações (inviável) — por isso o catálogo completo vai para o GA.

Complexidade: tempo O(L^n * n) | memória O(B * n) (B = tamanho do lote avaliado de uma vez).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from questao_01.modelo import InstanciaQ1, fitness, fitness_ingenuo


@dataclass
class ResultadoBF:
    niveis: np.ndarray
    fitness: float
    combinacoes: int
    viaveis: int


def n_combinacoes(inst: InstanciaQ1) -> int:
    return int(np.prod(inst.n_opcoes.astype(object)))


def brute_force(inst: InstanciaQ1, lote: int = 200_000, limite: int = 50_000_000) -> ResultadoBF:
    """Brute Force vetorizado: gera as combinações em lotes com np.unravel_index.

    A k-ésima combinação é o número k escrito na base mista (n_opcoes_1, ..., n_opcoes_n):
    isso evita guardar as N combinações na memória (só B por vez).
    """
    total = n_combinacoes(inst)
    if total > limite:
        raise ValueError(f"{total:,} combinações excedem o limite de {limite:,}; use o GA")
    melhor_f, melhor_x, viaveis = -np.inf, None, 0
    for ini in range(0, total, lote):
        k = np.arange(ini, min(ini + lote, total))
        genes = np.stack(np.unravel_index(k, tuple(inst.n_opcoes)), axis=1)   # (B, n)
        x = genes + inst.nivel_min                                             # índice do nível
        f = fitness(inst, x)
        viaveis += int(np.isfinite(f).sum())
        j = int(np.argmax(f))
        if f[j] > melhor_f:
            melhor_f, melhor_x = float(f[j]), x[j].copy()
    return ResultadoBF(melhor_x, melhor_f, total, viaveis)


def brute_force_ingenuo(inst: InstanciaQ1) -> ResultadoBF:
    """Versão 'antes': itertools.product + avaliação em Python puro, uma por vez."""
    opcoes = [range(int(a), len(inst.niveis)) for a in inst.nivel_min]
    melhor_f, melhor_x, total, viaveis = float("-inf"), None, 0, 0
    for comb in itertools.product(*opcoes):
        total += 1
        f = fitness_ingenuo(inst, comb)
        if f > float("-inf"):
            viaveis += 1
            if f > melhor_f:
                melhor_f, melhor_x = f, np.array(comb)
    return ResultadoBF(melhor_x, melhor_f, total, viaveis)
