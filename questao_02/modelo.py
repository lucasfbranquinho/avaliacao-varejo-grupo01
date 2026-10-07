"""Modelo econômico da Questão 2 (compartilhado por Monte Carlo, DP e Greedy).

Notação (usada em docs/complexidade.md)
  n = SKUs   T = semanas (12)   K = níveis de preço (6) + 1 ação "pausar"
  S = estoque inicial do SKU (tamanho do espaço de estados da DP)
  B = nº de faixas (quantis) da distribuição de demanda estimada pelo Monte Carlo
  N = nº de cenários Monte Carlo

Demanda semanal do SKU i no nível k, semana t, cenário w:
  lambda = D_i * m_k^(-e_i) * (r_k / r_0)^(-kappa) * choque_dem
  r_k = p_ik / pc_it   (preço relativo ao concorrente)   r_0 = P0_i / pc0_i
  e_i = elasticidade (x assimetria se m_k > 1, como na Q1)
  choque_dem = lognormal de média 1 e desvio sigma_demanda (fator comum de mercado + idiossincrático)
  pc_it = pc0_i * exp(x_it), x_it = rho * x_i,t-1 + sigma_pc * N(0,1)   (passeio AR(1) do concorrente)
  logística_t = l0_i * lognormal(média 1, sigma_logistica)                 (choque comum: frete/combustível)

Caixa semanal = sum_i vendas_i * (p_ik*(1-taxa_i) - custo_i - logistica_it) - custo dos destaques
(mesma convenção do notebook gerador: o custo do produto entra no caixa).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ParametrosQ2:
    kappa_concorrente: float = 1.5     # sensibilidade da demanda ao preço relativo ao concorrente
    rho_concorrente: float = 0.7       # persistência semanal do movimento do concorrente
    correlacao_demanda: float = 0.5    # fração da variância do choque de demanda que é comum a todos
    assimetria_aumento: float = 2.0    # elasticidade extra para preços acima do atual
    perda_obsolescencia: float = 0.10  # perda (fração do custo) do estoque que sobra na semana 12
    vender_com_prejuizo: bool = False  # False: níveis com contribuição esperada <= 0 são proibidos
    bins_demanda: int = 10             # B: faixas de quantis usadas pela DP
    alphas: tuple = (0.0, 0.25, 0.5, 1.0, 2.0)   # urgência de caixa testada pelo MC
    uplift_destaque: float = 0.25      # +25% de demanda no SKU destacado
    custo_destaque_frac: float = 0.03  # custo do destaque = 3% da receita esperada do SKU na semana
    custo_destaque_min: float = 150.0
    max_destaques_semana: int = 6
    retorno_minimo_destaque: float = 1.1  # margem de segurança: a validação MC mostrou que a
                                          # estimativa por faixas superestima o ganho em ~10%
    # cenário-surpresa
    concorrente_mult: float = 1.0
    logistica_mult: float = 1.0
    orcamento_mult: float = 1.0
    dividas_mult: float = 1.0
    demanda_mult: float = 1.0


@dataclass
class InstanciaQ2:
    skus: list[str]
    categorias: list[str]
    niveis: np.ndarray        # (K,) multiplicadores (sem a ação pausar)
    preco: np.ndarray         # (n, K)
    receita_unit: np.ndarray  # (n, K) p*(1-taxa) - custo  (falta descontar a logística da semana)
    estoque: np.ndarray       # (n,) int
    demanda_base: np.ndarray  # (n,)
    elasticidade: np.ndarray  # (n,)
    custo: np.ndarray         # (n,)
    logistica: np.ndarray     # (n,)
    pc0: np.ndarray           # (n,) preço do concorrente base
    r0: np.ndarray            # (n,) preço relativo de referência
    T: int
    semanas_divida: np.ndarray   # (3,) ex.: [4, 8, 12]
    dividas: np.ndarray          # (3,)
    caixa_inicial: float
    orcamento_semanal: float
    sigma_dem: float
    sigma_pc: float
    sigma_log: float
    limite_risco: float
    params: ParametrosQ2

    @property
    def n(self) -> int:
        return len(self.skus)

    @property
    def K(self) -> int:
        return len(self.niveis)

    @property
    def PAUSAR(self) -> int:
        return len(self.niveis)

    def pesos_urgencia(self, alpha: float) -> np.ndarray:
        """w_t = 1 + alpha * (fração da dívida que vence na semana t ou depois e ainda pode ser
        paga com o caixa gerado em t). Caixa cedo serve a mais compromissos => vale mais."""
        t = np.arange(1, self.T + 1)
        frac = np.array([(self.dividas[self.semanas_divida >= s]).sum() for s in t]) / self.dividas.sum()
        return 1.0 + alpha * frac


def construir_instancia(cat: pd.DataFrame, cfg: dict, params: ParametrosQ2 | None = None) -> InstanciaQ2:
    p = params or ParametrosQ2()
    niveis = np.asarray(cfg["niveis_preco_relativos"], float)
    if (niveis <= 0).any():
        raise ValueError("níveis de preço devem ser positivos")
    T = int(cfg["numero_semanas"])
    sem = np.array(sorted(int(k.split("_")[1]) for k in cfg["dividas"]))
    if sem.max() > T or sem.min() < 1:
        raise ValueError("semana de dívida fora do horizonte")
    div = np.array([cfg["dividas"][f"semana_{s}"] for s in sem], float) * p.dividas_mult
    p0 = cat["preco_atual"].to_numpy(float)
    taxa = cat["taxa_marketplace"].to_numpy(float)
    custo = cat["custo_unitario"].to_numpy(float)
    preco = p0[:, None] * niveis[None, :]
    pc0 = cat["preco_concorrente_base"].to_numpy(float)
    var = cfg["variabilidade"]
    return InstanciaQ2(
        skus=cat["sku"].tolist(), categorias=cat["categoria"].tolist(), niveis=niveis,
        preco=preco, receita_unit=preco * (1 - taxa[:, None]) - custo[:, None],
        estoque=cat["estoque_inicial"].to_numpy(int),
        demanda_base=cat["demanda_base_semanal"].to_numpy(float) * p.demanda_mult,
        elasticidade=cat["elasticidade"].to_numpy(float), custo=custo,
        logistica=cat["custo_logistico_base"].to_numpy(float) * p.logistica_mult,
        pc0=pc0 * p.concorrente_mult, r0=p0 / pc0,          # r0 usa o concorrente ORIGINAL (referência do cliente)
        T=T, semanas_divida=sem, dividas=div, caixa_inicial=float(cfg["caixa_inicial"]),
        orcamento_semanal=float(cfg["orcamento_promocional_semanal"]) * p.orcamento_mult,
        sigma_dem=var["sigma_demanda"], sigma_pc=var["sigma_preco_concorrente"],
        sigma_log=var["sigma_custo_logistico"], limite_risco=float(cfg["limite_risco_falha"]), params=p,
    )


def subinstancia(inst: InstanciaQ2, indices, escala_estoque: int = 1) -> InstanciaQ2:
    """Recorte por SKUs (e escala de estoque/demanda) — usado nos testes de escalabilidade."""
    idx = np.asarray(indices)
    from dataclasses import replace
    return replace(inst, skus=[inst.skus[i] for i in idx], categorias=[inst.categorias[i] for i in idx],
                   preco=inst.preco[idx], receita_unit=inst.receita_unit[idx],
                   estoque=inst.estoque[idx] * escala_estoque,
                   demanda_base=inst.demanda_base[idx] * escala_estoque,
                   elasticidade=inst.elasticidade[idx], custo=inst.custo[idx],
                   logistica=inst.logistica[idx], pc0=inst.pc0[idx], r0=inst.r0[idx])


def intensidade(inst: InstanciaQ2, k: np.ndarray, pc: np.ndarray) -> np.ndarray:
    """Demanda esperada (antes do choque) para o nível k (índices) e preço do concorrente pc.

    k e pc devem ser broadcast-compatíveis com o eixo de SKUs no final. k == PAUSAR => 0.
    """
    pausado = k >= inst.K
    kk = np.where(pausado, 0, k)
    m = inst.niveis[kk]
    preco = inst.preco[np.arange(inst.n), kk]          # indexação avançada: broadcast (..., n)
    eps = np.where(m > 1.0, inst.elasticidade * inst.params.assimetria_aumento, inst.elasticidade)
    rel = (preco / pc) / inst.r0
    lam = inst.demanda_base * m ** (-eps) * rel ** (-inst.params.kappa_concorrente)
    return np.where(pausado, 0.0, lam)
