"""Etapa C — Greedy: distribui o orçamento de incentivo comercial SOBRE a política do GA.

Entrada: a política de preços do GA (níveis por SKU). Para cada SKU há "degraus"
de cupom (5%, 10%, 15%, 20% do preço, pagos pela verba promocional). O cupom reduz
o preço efetivo para o cliente, então a demanda sobe pela elasticidade:
    d_ij = d_i(política) * (1 - r_j)^(-eps_i),   q_ij = min(E_i, floor(d_ij))
O preço recebido pela empresa não muda (quem paga o cupom é a verba), logo o
benefício de passar do degrau j-1 para j é
    beneficio = (q_ij - q_i,j-1) * (u_i + perda_obsolescencia * custo_i * fator_idade_i)
    custo     = r_j * p_i * q_ij - r_{j-1} * p_i * q_i,j-1
Regra gulosa: a cada passo compra o degrau com maior benefício por real investido
que ainda cabe no orçamento (fila de prioridade). Um SKU só pode subir um degrau
depois de comprar o anterior.

Complexidade: tempo O(n*J*log(n)) com heap (J = nº de degraus) | memória O(n*J).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

from questao_01.modelo import InstanciaQ1, avaliar, demanda

DEGRAUS_CUPOM = (0.05, 0.10, 0.15, 0.20)


@dataclass
class ResultadoGreedy:
    cupom: np.ndarray          # taxa de cupom escolhida por SKU (0 = nenhum)
    vendas: np.ndarray
    gasto: np.ndarray          # verba gasta por SKU
    orcamento: float
    caixa_antes: float
    caixa_depois: float
    fitness_antes: float
    fitness_depois: float
    tabela: pd.DataFrame


def _tabela_degraus(inst: InstanciaQ1, niveis: np.ndarray, degraus):
    """vendas q[i, j] e gasto acumulado g[i, j] para j = 0 (sem cupom) .. J."""
    taxas = np.concatenate([[0.0], np.asarray(degraus, float)])               # (J+1,)
    d = demanda(inst, niveis[None, :])[0]                                      # (n,)
    d_cupom = d[:, None] * (1 - taxas[None, :]) ** (-inst.elasticidade[:, None])
    q = np.minimum(inst.estoque[:, None], np.floor(d_cupom))                   # (n, J+1)
    preco = inst.preco[np.arange(inst.n), niveis]
    gasto = taxas[None, :] * preco[:, None] * q
    return taxas, q, gasto


def alocar_incentivos(inst: InstanciaQ1, niveis: np.ndarray, orcamento: float,
                      degraus=DEGRAUS_CUPOM, retorno_minimo: float = 0.0) -> ResultadoGreedy:
    if orcamento < 0:
        raise ValueError("orçamento promocional não pode ser negativo")
    niveis = np.asarray(niveis)
    taxas, q, gasto = _tabela_degraus(inst, niveis, degraus)
    u = inst.contrib[np.arange(inst.n), niveis]
    valor_unid = u + inst.params.perda_obsolescencia * inst.custo * inst.fator_idade

    def candidato(i, j):          # passo do degrau j-1 -> j do SKU i
        dq = q[i, j] - q[i, j - 1]
        dg = gasto[i, j] - gasto[i, j - 1]
        if dq <= 0 or dg <= 0:
            return None
        return (-(dq * valor_unid[i]) / dg, i, j, dg)

    heap = [c for i in range(inst.n) if (c := candidato(i, 1)) is not None]
    heapq.heapify(heap)
    escolhido = np.zeros(inst.n, dtype=int)
    restante = orcamento
    while heap:
        neg_ratio, i, j, dg = heapq.heappop(heap)
        if -neg_ratio <= retorno_minimo:
            break                                      # demais candidatos têm retorno ainda menor
        if dg > restante:
            continue                                   # não cabe: SKU encerra sua cadeia de degraus
        escolhido[i], restante = j, restante - dg
        if j + 1 < len(taxas) and (c := candidato(i, j + 1)) is not None:
            heapq.heappush(heap, c)

    linhas = np.arange(inst.n)
    q_dep, g_dep = q[linhas, escolhido], gasto[linhas, escolhido]
    antes = avaliar(inst, niveis)
    caixa_ant, fit_ant = float(antes["caixa"][0]), float(antes["fitness"][0])
    caixa_dep = float((q_dep * u).sum())
    fit_dep = caixa_dep - inst.params.perda_obsolescencia * float(
        ((inst.estoque - q_dep) * inst.custo * inst.fator_idade).sum())

    tabela = pd.DataFrame({
        "sku": inst.skus, "categoria": inst.categorias,
        "multiplicador": inst.niveis[niveis], "preco": inst.preco[linhas, niveis].round(2),
        "margem_pct": (100 * inst.margem_pct[linhas, niveis]).round(1),
        "estoque": inst.estoque.astype(int),
        "vendas_antes": q[:, 0].astype(int), "caixa_antes": (q[:, 0] * u).round(2),
        "cupom_pct": (100 * taxas[escolhido]).round(0), "verba_gasta": g_dep.round(2),
        "vendas_depois": q_dep.astype(int), "caixa_depois": (q_dep * u).round(2),
        "estoque_restante": (inst.estoque - q_dep).astype(int),
    })
    tabela["ganho_caixa"] = (tabela["caixa_depois"] - tabela["caixa_antes"]).round(2)
    return ResultadoGreedy(taxas[escolhido], q_dep, g_dep, orcamento, caixa_ant, caixa_dep,
                           fit_ant, fit_dep, tabela)


def alocacao_uniforme(inst: InstanciaQ1, niveis: np.ndarray, orcamento: float,
                      degraus=DEGRAUS_CUPOM) -> tuple[float, float, float]:
    """Linha de base para comparar com o Greedy: mesmo cupom para TODOS os SKUs.

    Retorna (taxa, caixa, gasto) do maior cupom uniforme que cabe no orçamento.
    """
    niveis = np.asarray(niveis)
    taxas, q, gasto = _tabela_degraus(inst, niveis, degraus)
    u = inst.contrib[np.arange(inst.n), niveis]
    melhor = (0.0, float((q[:, 0] * u).sum()), 0.0)
    for j in range(1, len(taxas)):
        if gasto[:, j].sum() <= orcamento:
            melhor = (float(taxas[j]), float((q[:, j] * u).sum()), float(gasto[:, j].sum()))
    return melhor
