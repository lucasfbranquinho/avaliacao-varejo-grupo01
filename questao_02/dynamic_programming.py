"""Etapa B — Programação Dinâmica estocástica (Bottom-Up) por SKU.

FORMULAÇÃO
  Estado:     (t, s)  t = semana (0..T-1), s = estoque restante do SKU (0..S)
  Decisão:    k em {nível 0.82, ..., 1.12, PAUSAR}
  Incerteza:  demanda d_b com probabilidade 1/B (faixas de quantis estimadas pelo Monte Carlo)
  Transição:  s' = s - min(s, d_b)
  Recompensa: w_t * min(s, d_b) * (p_k*(1-taxa) - custo - E[logística_t])
              (w_t = peso de urgência: caixa que chega antes dos vencimentos vale mais)
  Terminal:   V_T(s) = -perda_obsolescencia * custo * s   (estoque que sobra perde valor)
  Restrição:  níveis com contribuição esperada <= 0 são proibidos (não vender abaixo do custo);
              se nenhum nível é lucrativo, a única ação possível é PAUSAR.
  Bellman:    V_t(s) = max_k  (1/B) * sum_b [ w_t*R(s,k,d_b) + V_{t+1}(s - min(s, d_b)) ]

REUTILIZAÇÃO DE SUBPROBLEMAS
  O vetor V_{t+1}(.) é calculado UMA vez e reutilizado por todos os (s, k, b) da semana t:
  cada valor V_{t+1}(s') é consultado por muitos pares (s, d_b) diferentes que levam ao mesmo s'.
  Sem a tabela, a árvore de decisões teria (K*B)^T caminhos por estado inicial
  (7*10)^12 ~ 1,4e22; com a tabela são T*(S+1)*K*B operações (~0,4 milhão por SKU).

Por que a DP é por SKU: o estado conjunto (s_1..s_n) teria prod(S_i+1) ~ 10^50 estados.
O acoplamento entre SKUs (o caixa comum que paga as dívidas) entra pelos pesos w_t,
calibrados pelo Monte Carlo (alpha), e é verificado na validação final.

Complexidade: tempo O(n*T*S*K*B) | memória O(n*T*S) (tabela de política) + O(S) por estágio.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from questao_02.modelo import InstanciaQ2
from questao_02.monte_carlo import EstatisticasMC


@dataclass
class ResultadoDP:
    politica: np.ndarray     # (n, T, Smax+1) ação ótima k*(t, s)
    valor: list              # por SKU: array (T+1, S_i+1) com V_t(s)
    alpha: float
    pesos: np.ndarray        # (T,)


def _recompensas(inst: InstanciaQ2, est: EstatisticasMC, i: int):
    """Contribuição unitária esperada u[t, k] (k = PAUSAR -> 0) e demanda inteira d[t, k, b]."""
    u = inst.receita_unit[i][None, :] - est.logistica_media[:, i][:, None]          # (T, K)
    d = np.rint(est.demanda_bins[:, i, :, :]).astype(int)                           # (T, K, B)
    return u, d


def resolver_sku(inst: InstanciaQ2, est: EstatisticasMC, i: int, pesos: np.ndarray):
    """Bottom-Up vetorizado sobre o estoque s. Retorna (V (T+1, S+1), politica (T, S+1))."""
    S, T, K = int(inst.estoque[i]), inst.T, inst.K
    B = est.demanda_bins.shape[-1]
    u, d = _recompensas(inst, est, i)
    s = np.arange(S + 1)
    V = np.zeros((T + 1, S + 1))
    V[T] = -inst.params.perda_obsolescencia * inst.custo[i] * s
    pol = np.zeros((T, S + 1), dtype=np.int16)
    for t in range(T - 1, -1, -1):
        q = np.empty((K + 1, S + 1))
        for k in range(K):
            vend = np.minimum(s[:, None], d[t, k][None, :])                          # (S+1, B)
            q[k] = (pesos[t] * vend * u[t, k] + V[t + 1][s[:, None] - vend]).sum(axis=1) / B
            if u[t, k] <= 0 and not inst.params.vender_com_prejuizo:
                q[k, 1:] = -np.inf                    # proibido vender abaixo do custo (s=0 é indiferente)
        q[K] = V[t + 1]                                                               # PAUSAR
        pol[t] = np.argmax(q, axis=0)
        V[t] = q[pol[t], s]
    return V, pol


def programacao_dinamica(inst: InstanciaQ2, est: EstatisticasMC, alpha: float = 0.0) -> ResultadoDP:
    pesos = inst.pesos_urgencia(alpha)
    smax = int(inst.estoque.max())
    politica = np.full((inst.n, inst.T, smax + 1), inst.PAUSAR, dtype=np.int16)
    valores = []
    for i in range(inst.n):
        V, pol = resolver_sku(inst, est, i, pesos)
        politica[i, :, : pol.shape[1]] = pol
        valores.append(V)
    return ResultadoDP(politica, valores, alpha, pesos)


def resolver_sku_top_down(inst: InstanciaQ2, est: EstatisticasMC, i: int, pesos: np.ndarray):
    """Mesma recursão em versão Top-Down com memoization (lru_cache).

    Usada nos testes para provar que Bottom-Up e Top-Down dão o mesmo V_0(S) e para
    contar quantas vezes um subproblema é REUTILIZADO (hits do cache).
    """
    u, d = _recompensas(inst, est, i)
    T, K = inst.T, inst.K
    B = d.shape[-1]
    perda = inst.params.perda_obsolescencia * inst.custo[i]

    @lru_cache(maxsize=None)
    def V(t: int, s: int) -> float:
        if t == T:
            return -perda * s
        melhor = V(t + 1, s)                                  # PAUSAR
        for k in range(K):
            if u[t, k] <= 0 and not inst.params.vender_com_prejuizo and s > 0:
                continue
            tot = 0.0
            for b in range(B):
                v = min(s, int(d[t, k, b]))
                tot += pesos[t] * v * u[t, k] + V(t + 1, s - v)
            melhor = max(melhor, tot / B)
        return melhor

    valor = V(0, int(inst.estoque[i]))
    return valor, V.cache_info()


def trajetoria_esperada(inst: InstanciaQ2, est: EstatisticasMC, dp: ResultadoDP) -> pd.DataFrame:
    """Reconstrói o caminho ótimo: parte de S_i e segue k*(t, s) com a venda ESPERADA
    (média sobre as faixas) arredondada. É o 'plano' que orienta o Greedy."""
    linhas = []
    for i in range(inst.n):
        s = int(inst.estoque[i])
        for t in range(inst.T):
            k = int(dp.politica[i, t, s])
            if k == inst.PAUSAR:
                venda, mult, preco = 0, np.nan, np.nan
            else:
                d = np.rint(est.demanda_bins[t, i, k])
                venda = int(np.rint(np.minimum(s, d).mean()))
                mult, preco = inst.niveis[k], inst.preco[i, k]
            linhas.append({"sku": inst.skus[i], "semana": t + 1, "estoque_inicio": s, "acao": k,
                           "multiplicador": mult, "preco": preco, "venda_esperada": venda,
                           "estoque_fim": s - venda})
            s -= venda
    return pd.DataFrame(linhas)
