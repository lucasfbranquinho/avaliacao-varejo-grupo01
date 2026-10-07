"""Modelo econômico da Questão 1 — compartilhado por Brute Force, GA e Greedy.

Notação (usada também em docs/complexidade.md)
---------------------------------------------
n  = número de SKUs ativos            L  = número de níveis de preço (7)
m_k = multiplicador do nível k        P0_i = preco_atual do SKU i
p_ik = P0_i * m_k                     (preço praticado)
u_ik = p_ik*(1 - taxa_i) - custo_i - logistica_i   (contribuição unitária = caixa por unidade)

Demanda em 90 dias (elasticidade constante + canibalização na categoria):
    d_i = D_i * m_i^(-e_i) * (m_i / mbar_c(i))^(-gama)
    e_i = eps_i                se m_i <= 1   (desconto)
    e_i = eps_i * assimetria   se m_i  > 1   (aumento acima do preço de referência)
A assimetria modela o efeito de preço de referência: no canal online o cliente
compara preços, e um aumento perde mais demanda do que um desconto equivalente
ganha. Sem ela, com margens finas, o ótimo seria trivialmente "tudo em +10%".
onde mbar_c é a média dos multiplicadores dos SKUs da mesma categoria.
Se todos os SKUs da categoria descontam igualmente, ninguém "rouba" demanda
do vizinho; se só um desconta, ele ganha participação. É esse termo que
acopla os SKUs e torna o problema combinatório (não separável), justificando
Brute Force/GA em vez de otimizar cada SKU isoladamente.

Vendas: q_i = min(estoque_i, floor(d_i))
Caixa operacional: C = sum_i q_i * u_ik
Penalidade de giro (estoque velho parado): G = perda_obsolescencia * sum_i (E_i - q_i) * custo_i * min(1, idade_i/360)
(perda esperada de valor do estoque que sobra após 90 dias; cresce com a idade)
Fitness = C - G

Restrições
- piso de margem: u_ik / p_ik >= margem_minima  (níveis que violam são proibidos)
- estoque: q_i <= E_i (pela própria fórmula de q)
- anti-desconto indiscriminado: no máximo floor(max_frac_desconto * n) SKUs com m < 1
SKUs sem nenhum nível que respeite o piso são SUSPENSOS (retirados da política).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ParametrosQ1:
    gama_canibalizacao: float = 0.5   # força da troca de demanda entre SKUs da categoria
    assimetria_aumento: float = 2.0   # elasticidade de aumento = elasticidade * assimetria
    perda_obsolescencia: float = 0.25 # perda de valor do estoque residual com 1 ano (fração do custo)
    max_frac_desconto: float = 0.40   # fração máxima de SKUs com preço abaixo do atual
    meta_mult: float = 1.0            # cenário-surpresa: meta de caixa * meta_mult
    logistica_mult: float = 1.0       # cenário-surpresa: custo logístico * mult
    orcamento_mult: float = 1.0       # cenário-surpresa: orçamento promocional * mult


@dataclass
class InstanciaQ1:
    """Arrays NumPy pré-calculados para avaliação rápida (todas as métricas por SKU x nível)."""
    skus: list[str]
    categorias: list[str]
    niveis: np.ndarray            # (L,)
    preco: np.ndarray             # (n, L)
    contrib: np.ndarray           # (n, L)  u_ik
    margem_pct: np.ndarray        # (n, L)
    nivel_min: np.ndarray         # (n,) menor nível que respeita o piso de margem
    n_opcoes: np.ndarray          # (n,) quantos níveis permitidos (L - nivel_min)
    demanda_base: np.ndarray      # (n,)
    elasticidade: np.ndarray      # (n,)
    estoque: np.ndarray           # (n,)
    custo: np.ndarray             # (n,)
    fator_idade: np.ndarray       # (n,)
    onehot_cat: np.ndarray        # (n, n_cat)
    tam_cat: np.ndarray           # (n_cat,)
    max_descontos: int
    params: ParametrosQ1
    suspensos: list[str] = field(default_factory=list)
    idx_nivel_atual: int = 4      # índice do multiplicador 1.00

    @property
    def n(self) -> int:
        return len(self.skus)

    def nivel_padrao(self) -> np.ndarray:
        """Política 'não fazer nada': preço atual (1.00) ou o menor nível viável acima dele."""
        return np.maximum(self.nivel_min, self.idx_nivel_atual)


def construir_instancia(cat: pd.DataFrame, cfg: dict, params: ParametrosQ1 | None = None) -> InstanciaQ1:
    params = params or ParametrosQ1()
    niveis = np.asarray(cfg["niveis_preco_relativos"], dtype=float)
    if len(niveis) == 0 or (niveis <= 0).any():
        raise ValueError("níveis de preço devem ser positivos")
    if not 0 <= cfg["margem_minima_percentual"] < 1:
        raise ValueError("margem mínima fora de [0, 1)")

    p0 = cat["preco_atual"].to_numpy(float)
    taxa = cat["taxa_marketplace"].to_numpy(float)
    custo = cat["custo_unitario"].to_numpy(float)
    logist = cat["custo_logistico"].to_numpy(float) * params.logistica_mult

    preco = p0[:, None] * niveis[None, :]
    contrib = preco * (1 - taxa[:, None]) - custo[:, None] - logist[:, None]
    margem = contrib / preco
    viavel = margem >= cfg["margem_minima_percentual"]
    # margem% cresce com o preço, então os níveis viáveis formam um sufixo [nivel_min, L)
    nivel_min = np.where(viavel.any(axis=1), viavel.argmax(axis=1), len(niveis))

    ativos = nivel_min < len(niveis)
    suspensos = cat.loc[~ativos, "sku"].tolist()
    sel = np.flatnonzero(ativos)

    categorias = cat["categoria"].to_numpy()[sel]
    cats_unicas, cod = np.unique(categorias, return_inverse=True)
    onehot = np.zeros((len(sel), len(cats_unicas)))
    onehot[np.arange(len(sel)), cod] = 1.0

    idx_atual = int(np.argmin(np.abs(niveis - 1.0)))
    return InstanciaQ1(
        skus=cat["sku"].to_numpy()[sel].tolist(),
        categorias=categorias.tolist(),
        niveis=niveis,
        preco=preco[sel],
        contrib=contrib[sel],
        margem_pct=margem[sel],
        nivel_min=nivel_min[sel],
        n_opcoes=len(niveis) - nivel_min[sel],
        demanda_base=cat["demanda_base_90d"].to_numpy(float)[sel],
        elasticidade=cat["elasticidade"].to_numpy(float)[sel],
        estoque=cat["estoque"].to_numpy(float)[sel],
        custo=custo[sel],
        fator_idade=np.minimum(1.0, cat["idade_estoque_dias"].to_numpy(float)[sel] / 360),
        onehot_cat=onehot,
        tam_cat=onehot.sum(axis=0),
        max_descontos=int(np.floor(params.max_frac_desconto * len(sel))),
        params=params,
        suspensos=suspensos,
        idx_nivel_atual=idx_atual,
    )


def subinstancia(inst: InstanciaQ1, indices) -> InstanciaQ1:
    """Recorta a instância (usado no subconjunto do Brute Force e nos testes de escala)."""
    idx = np.asarray(indices)
    oh = inst.onehot_cat[idx]
    usadas = oh.sum(axis=0) > 0
    oh = oh[:, usadas]
    return InstanciaQ1(
        skus=[inst.skus[i] for i in idx], categorias=[inst.categorias[i] for i in idx],
        niveis=inst.niveis, preco=inst.preco[idx], contrib=inst.contrib[idx],
        margem_pct=inst.margem_pct[idx], nivel_min=inst.nivel_min[idx], n_opcoes=inst.n_opcoes[idx],
        demanda_base=inst.demanda_base[idx], elasticidade=inst.elasticidade[idx],
        estoque=inst.estoque[idx], custo=inst.custo[idx], fator_idade=inst.fator_idade[idx],
        onehot_cat=oh, tam_cat=oh.sum(axis=0),
        max_descontos=int(np.floor(inst.params.max_frac_desconto * len(idx))),
        params=inst.params, suspensos=[], idx_nivel_atual=inst.idx_nivel_atual,
    )


# ---------------------------------------------------------------------------
# Avaliação vetorizada: recebe uma MATRIZ de políticas (P x n níveis) e
# devolve o fitness das P políticas de uma vez. É o coração do custo de
# BF e GA: O(P * n) operações (o produto pela one-hot custa O(P * n * n_cat)).
# ---------------------------------------------------------------------------

def vendas(inst: InstanciaQ1, niveis_idx: np.ndarray) -> np.ndarray:
    m = inst.niveis[niveis_idx]                                   # (P, n)
    soma_cat = m @ inst.onehot_cat                                # (P, n_cat)
    media_cat = (soma_cat / inst.tam_cat) @ inst.onehot_cat.T     # (P, n)
    eps = np.where(m > 1.0, inst.elasticidade * inst.params.assimetria_aumento, inst.elasticidade)
    d = (inst.demanda_base * m ** (-eps)
         * (m / media_cat) ** (-inst.params.gama_canibalizacao))
    return np.minimum(inst.estoque, np.floor(d))


def avaliar(inst: InstanciaQ1, niveis_idx: np.ndarray) -> dict:
    """Métricas completas de uma ou várias políticas. niveis_idx: (n,) ou (P, n)."""
    x = np.atleast_2d(niveis_idx)
    q = vendas(inst, x)
    linhas = np.arange(inst.n)
    u = inst.contrib[linhas, x]                                   # (P, n)
    receita = (q * inst.preco[linhas, x]).sum(axis=1)
    caixa = (q * u).sum(axis=1)
    resid = inst.estoque - q
    pen_giro = inst.params.perda_obsolescencia * (resid * inst.custo * inst.fator_idade).sum(axis=1)
    n_desc = (inst.niveis[x] < 1.0).sum(axis=1)
    return {
        "caixa": caixa, "receita": receita, "pen_giro": pen_giro,
        "fitness": caixa - pen_giro, "n_descontos": n_desc,
        "viavel": (n_desc <= inst.max_descontos) & (x >= inst.nivel_min).all(axis=1),
        "vendas": q, "residual": resid,
    }


def fitness(inst: InstanciaQ1, niveis_idx: np.ndarray) -> np.ndarray:
    r = avaliar(inst, niveis_idx)
    return np.where(r["viavel"], r["fitness"], -np.inf)


def fitness_ingenuo(inst: InstanciaQ1, niveis_idx) -> float:
    """Versão 'antes da otimização': laços Python puros, uma política por vez.

    Mantida apenas para a demonstração antes/depois em questao_01/benchmark.py
    (mesmo resultado numérico de fitness(), ordens de grandeza mais lenta).
    """
    n = inst.n
    m = [inst.niveis[niveis_idx[i]] for i in range(n)]
    n_desc = sum(1 for v in m if v < 1.0)
    if n_desc > inst.max_descontos or any(niveis_idx[i] < inst.nivel_min[i] for i in range(n)):
        return float("-inf")
    total = 0.0
    for i in range(n):
        mesma = [m[j] for j in range(n) if inst.categorias[j] == inst.categorias[i]]
        media = sum(mesma) / len(mesma)
        eps = inst.elasticidade[i] * (inst.params.assimetria_aumento if m[i] > 1.0 else 1.0)
        d = (inst.demanda_base[i] * m[i] ** (-eps)
             * (m[i] / media) ** (-inst.params.gama_canibalizacao))
        q = min(inst.estoque[i], np.floor(d))
        total += q * inst.contrib[i, niveis_idx[i]]
        total -= inst.params.perda_obsolescencia * (inst.estoque[i] - q) * inst.custo[i] * inst.fator_idade[i]
    return total
