"""Etapas A e D — Monte Carlo: estimar a demanda/caixa sob incerteza e validar o plano final.

Três fontes de incerteza (enunciado 2.4):
  1. demanda do consumidor   -> choque lognormal (fator comum de mercado + idiossincrático)
  2. preço do concorrente     -> passeio AR(1) em log, por SKU
  3. custo logístico          -> choque lognormal semanal comum a todos os SKUs (frete/combustível)

Construção (A): N_c = 2000 cenários com seed s geram, para cada (semana, SKU, nível), a
distribuição empírica da demanda, resumida em B faixas de quantis -> entrada da DP.
Validação (D): N_v = 5000 cenários NOVOS (seed diferente) simulam o plano completo
(política DP em malha fechada + destaques do Greedy) e medem o risco de não pagar as dívidas.

Complexidade:
  estimar_estatisticas: tempo O(N*T*n*K*log N) (ordenação p/ quantis) | memória O(N*T*n*K)
  simular:              tempo O(N*T*n)                                | memória O(N*n + N*T)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from questao_02.modelo import InstanciaQ2, intensidade

SEED_OFFSET_VALIDACAO = 7777     # garante cenários de validação independentes dos de construção


@dataclass
class Cenarios:
    choque_dem: np.ndarray   # (N, T, n) multiplicador de demanda (média 1)
    pc: np.ndarray           # (N, T, n) preço do concorrente
    logistica: np.ndarray    # (N, T, n) custo logístico por unidade
    seed: int

    @property
    def N(self) -> int:
        return self.choque_dem.shape[0]


def gerar_cenarios(inst: InstanciaQ2, N: int, seed: int) -> Cenarios:
    if N < 1:
        raise ValueError("número de cenários deve ser positivo")
    rng = np.random.default_rng(seed)
    p = inst.params
    T, n = inst.T, inst.n
    # 1) demanda: Z = sqrt(rho)*Z_comum + sqrt(1-rho)*Z_sku ; choque = exp(s*Z - s^2/2) (média 1)
    z = (np.sqrt(p.correlacao_demanda) * rng.standard_normal((N, T, 1))
         + np.sqrt(1 - p.correlacao_demanda) * rng.standard_normal((N, T, n)))
    choque = np.exp(inst.sigma_dem * z - inst.sigma_dem ** 2 / 2)
    # 2) concorrente: x_t = rho*x_{t-1} + sigma*eps
    eps = rng.standard_normal((N, T, n)) * inst.sigma_pc
    x = np.zeros((N, T, n))
    x[:, 0] = eps[:, 0]
    for t in range(1, T):
        x[:, t] = p.rho_concorrente * x[:, t - 1] + eps[:, t]
    pc = inst.pc0 * np.exp(x)
    # 3) logística: choque comum semanal
    zl = rng.standard_normal((N, T, 1))
    logist = inst.logistica * np.exp(inst.sigma_log * zl - inst.sigma_log ** 2 / 2)
    return Cenarios(choque, pc, logist, seed)


@dataclass
class EstatisticasMC:
    demanda_bins: np.ndarray       # (T, n, K, B) valor médio de cada faixa de quantil (prob. 1/B cada)
    logistica_media: np.ndarray    # (T, n)
    resumo: pd.DataFrame           # por SKU x nível: média, p5, p95 da demanda e caixa esperado/semana


def estimar_estatisticas(inst: InstanciaQ2, cen: Cenarios) -> EstatisticasMC:
    B = inst.params.bins_demanda
    N = cen.N - cen.N % B                       # usa múltiplo de B para faixas de mesmo tamanho
    T, n, K = inst.T, inst.n, inst.K
    bins = np.empty((T, n, K, B))
    linhas = []
    log_med = cen.logistica[:N].mean(axis=0)    # (T, n)
    for k in range(K):
        kk = np.full((N, T, n), k)
        d = intensidade(inst, kk, cen.pc[:N]) * cen.choque_dem[:N]      # (N, T, n)
        d.sort(axis=0)
        bins[:, :, k, :] = d.reshape(B, N // B, T, n).mean(axis=1).transpose(1, 2, 0)
        unit = inst.receita_unit[:, k] - log_med.mean(axis=0)            # (n,)
        for i in range(n):
            amostra = d[:, :, i]
            linhas.append({
                "sku": inst.skus[i], "multiplicador": inst.niveis[k], "preco": round(inst.preco[i, k], 2),
                "demanda_media": amostra.mean(), "demanda_p5": np.percentile(amostra, 5),
                "demanda_p95": np.percentile(amostra, 95),
                "caixa_unit_esperado": unit[i],
                "caixa_semanal_esperado_sem_limite": amostra.mean() * unit[i],
            })
    return EstatisticasMC(bins, log_med, pd.DataFrame(linhas).round(3))


@dataclass
class ResultadoSimulacao:
    caixa_antes_pagamento: np.ndarray   # (N, D) caixa disponível em cada vencimento
    pago: np.ndarray                    # (N, D) bool
    caixa_final: np.ndarray             # (N,) após pagar (ou tentar pagar) as dívidas
    caixa_semanal: np.ndarray           # (N, T) caixa acumulado ao fim de cada semana (após dívidas)
    receita: np.ndarray                 # (N,)
    contribuicao: np.ndarray            # (N,) caixa operacional gerado (sem caixa inicial, com destaques)
    gasto_destaques: float
    estoque_final: np.ndarray           # (N, n)
    estoque_semana: np.ndarray          # (N, T+1, n)
    nivel_semana: np.ndarray            # (N, T, n) ação escolhida (K = pausado)

    @property
    def prob_pagamento(self) -> np.ndarray:
        return self.pago.mean(axis=0)

    @property
    def prob_pagar_tudo(self) -> float:
        return float(self.pago.all(axis=1).mean())


def simular(inst: InstanciaQ2, cen: Cenarios, politica: np.ndarray,
            destaques: np.ndarray | None = None, custo_destaque: np.ndarray | None = None) -> ResultadoSimulacao:
    """Simula a política em MALHA FECHADA: a cada semana o nível vem da tabela da DP
    consultada com o estoque REALIZADO naquele cenário (politica[i, t, s])."""
    N, T, n = cen.N, inst.T, inst.n
    if destaques is None:
        destaques = np.zeros((T, n), bool)
        custo_destaque = np.zeros((T, n))
    sku = np.arange(n)
    s = np.broadcast_to(inst.estoque, (N, n)).copy()
    caixa = np.full(N, inst.caixa_inicial)
    receita = np.zeros(N)
    contrib = np.zeros(N)
    D = len(inst.semanas_divida)
    antes = np.zeros((N, D))
    pago = np.zeros((N, D), bool)
    cx_sem = np.zeros((N, T))
    est_sem = np.zeros((N, T + 1, n), dtype=np.int32)
    niv_sem = np.zeros((N, T, n), dtype=np.int8)
    est_sem[:, 0] = s
    gasto_dest = float((destaques * custo_destaque).sum())
    j = 0
    for t in range(T):
        k = politica[sku, t, s]                                         # (N, n)
        lam = intensidade(inst, k, cen.pc[:, t]) * cen.choque_dem[:, t]
        lam = lam * (1 + inst.params.uplift_destaque * destaques[t])
        vendas = np.minimum(s, np.rint(lam).astype(s.dtype))
        kk = np.minimum(k, inst.K - 1)
        unit = inst.receita_unit[sku, kk] - cen.logistica[:, t]
        fluxo = (vendas * unit).sum(axis=1) - (destaques[t] * custo_destaque[t]).sum()
        caixa += fluxo
        contrib += fluxo
        receita += (vendas * inst.preco[sku, kk]).sum(axis=1)
        s = s - vendas
        est_sem[:, t + 1] = s
        niv_sem[:, t] = k
        if j < D and t + 1 == inst.semanas_divida[j]:
            antes[:, j] = caixa
            pago[:, j] = caixa >= inst.dividas[j] - 1e-9
            caixa -= inst.dividas[j]
            j += 1
        cx_sem[:, t] = caixa
    return ResultadoSimulacao(antes, pago, caixa, cx_sem, receita, contrib, gasto_dest, s, est_sem, niv_sem)


def politica_fixa(inst: InstanciaQ2, nivel: int) -> np.ndarray:
    """Linha de base: mesmo nível para todas as semanas e estoques (ex.: preço atual)."""
    return np.full((inst.n, inst.T, int(inst.estoque.max()) + 1), nivel, dtype=np.int16)
