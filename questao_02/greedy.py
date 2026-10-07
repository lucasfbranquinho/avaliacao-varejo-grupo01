"""Etapa C — Greedy semanal: quais SKUs recebem o recurso promocional escasso (destaque/mídia).

Entrada: o PLANO da DP (estoque e nível esperados por semana) e a função de valor V da DP.
Um destaque aumenta a demanda do SKU em +uplift naquela semana e custa
    custo_it = max(custo_min, frac * receita esperada do SKU na semana)
O benefício é medido com a própria função de valor da DP, para não "comprar" vendas que
só seriam antecipadas (um SKU que vai esgotar de qualquer jeito ganha pouco com destaque):
    beneficio_it = E_b[ w_t*u*v' + V_{t+1}(s - v') ] - E_b[ w_t*u*v + V_{t+1}(s - v) ]
    v = min(s, d_b),  v' = min(s, round(d_b*(1+uplift)))
Regra gulosa, a cada semana: ordenar por beneficio/(w_t*custo) e escolher enquanto
houver orçamento semanal, vagas (max_destaques_semana) e razão > retorno_minimo (1,1).
A margem de 10% acima de 1 vem da validação: o Monte Carlo independente mostrou que a
estimativa por faixas de quantis superestima o ganho real do destaque em ~10-20% em SKUs
de giro baixo (arredondamento da demanda), e com razão 1,0 o Greedy chegava a PERDER caixa.

Complexidade: tempo O(T * n * (B + log n)) | memória O(T * n).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from questao_02.dynamic_programming import ResultadoDP
from questao_02.modelo import InstanciaQ2
from questao_02.monte_carlo import EstatisticasMC


@dataclass
class ResultadoGreedyQ2:
    destaques: np.ndarray     # (T, n) bool
    custo: np.ndarray         # (T, n) custo de cada destaque possível
    escolhas: pd.DataFrame    # candidatos avaliados e decisão
    gasto_semanal: np.ndarray # (T,)


def alocar_destaques(inst: InstanciaQ2, est: EstatisticasMC, dp: ResultadoDP,
                     plano: pd.DataFrame, orcamento_semanal: float | None = None) -> ResultadoGreedyQ2:
    orc = inst.orcamento_semanal if orcamento_semanal is None else orcamento_semanal
    if orc < 0:
        raise ValueError("orçamento semanal não pode ser negativo")
    p = inst.params
    T, n = inst.T, inst.n
    destaques = np.zeros((T, n), bool)
    custo = np.zeros((T, n))
    gasto = np.zeros(T)
    plano_idx = plano.set_index(["sku", "semana"])
    linhas = []
    for t in range(T):
        w = dp.pesos[t]
        cands = []
        for i in range(n):
            lin = plano_idx.loc[(inst.skus[i], t + 1)]
            k, s = int(lin["acao"]), int(lin["estoque_inicio"])
            if k == inst.PAUSAR or s == 0:
                continue
            V1 = dp.valor[i][t + 1]
            u = inst.receita_unit[i, k] - est.logistica_media[t, i]
            d = est.demanda_bins[t, i, k]
            v = np.minimum(s, np.rint(d)).astype(int)
            v2 = np.minimum(s, np.rint(d * (1 + p.uplift_destaque))).astype(int)
            ganho = float(np.mean(w * u * v2 + V1[s - v2]) - np.mean(w * u * v + V1[s - v]))
            c = max(p.custo_destaque_min, p.custo_destaque_frac * inst.preco[i, k] * v.mean())
            custo[t, i] = c
            cands.append((ganho / (w * c), i, ganho, c))
        cands.sort(reverse=True)                 # maior benefício por real primeiro
        restante, vagas = orc, p.max_destaques_semana
        for razao, i, ganho, c in cands:
            escolhe = razao > p.retorno_minimo_destaque and c <= restante and vagas > 0
            if escolhe:
                destaques[t, i] = True
                restante -= c
                vagas -= 1
                gasto[t] += c
            linhas.append({"semana": t + 1, "sku": inst.skus[i], "beneficio_ponderado": round(ganho, 2),
                           "custo": round(c, 2), "razao_beneficio_custo": round(razao, 3),
                           "escolhido": escolhe})
    return ResultadoGreedyQ2(destaques, custo, pd.DataFrame(linhas), gasto)
