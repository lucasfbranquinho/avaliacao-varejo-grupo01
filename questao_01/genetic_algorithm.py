"""Etapa B — Algoritmo Genético para o catálogo completo.

Representação: cromossomo = vetor de n genes inteiros; gene i em [0, L - nivel_min_i)
indica qual dos níveis PERMITIDOS (que respeitam o piso de margem) o SKU i usa.
Assim nenhum indivíduo viola o piso de margem por construção.

Operadores (justificativas em docs/decisoes.md):
- inicialização: aleatória + 1 indivíduo semente com a política atual (preço 1.00);
- seleção: torneio de tamanho k (pressão seletiva ajustável, não depende da escala do fitness);
- crossover: uniforme (padrão) ou um ponto, com probabilidade pc;
- mutação: reinicialização aleatória do gene com probabilidade pm por gene (padrão 1/n);
- reparo: se a política tem mais descontos que o permitido, SKUs excedentes voltam ao preço atual;
- elitismo: os e melhores passam intactos;
- parada: geracoes_max OU 'paciencia' gerações sem melhora do melhor fitness.

Complexidade: tempo O(G * P * n) | memória O(P * n + G) (população + histórico).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from questao_01.modelo import InstanciaQ1, fitness


@dataclass
class ConfigGA:
    nome: str = "A"
    populacao: int = 100
    geracoes_max: int = 300
    paciencia: int = 60
    torneio: int = 3
    taxa_crossover: float = 0.9
    tipo_crossover: str = "uniforme"     # "uniforme" | "um_ponto"
    taxa_mutacao: float | None = None    # None => 1/n
    elitismo: int = 2
    semente_heuristica: bool = True

    def validar(self) -> None:
        if self.populacao < 2 or self.elitismo >= self.populacao:
            raise ValueError("população deve ser >= 2 e maior que o elitismo")
        if not 0 <= self.taxa_crossover <= 1:
            raise ValueError("taxa_crossover fora de [0, 1]")
        if self.taxa_mutacao is not None and not 0 <= self.taxa_mutacao <= 1:
            raise ValueError("taxa_mutacao fora de [0, 1]")
        if self.tipo_crossover not in ("uniforme", "um_ponto"):
            raise ValueError("tipo_crossover inválido")


# Configurações comparadas no relatório (entrega 1.4.5)
CONFIG_A = ConfigGA(nome="A (P=100, torneio 3, uniforme, pm=1/n, elite 2)")
CONFIG_B = ConfigGA(nome="B (P=30, torneio 2, um ponto, pm=0.10, elite 1)", populacao=30,
                    torneio=2, tipo_crossover="um_ponto", taxa_mutacao=0.10, elitismo=1)


@dataclass
class ResultadoGA:
    niveis: np.ndarray
    fitness: float
    historico_melhor: list[float] = field(default_factory=list)
    historico_media: list[float] = field(default_factory=list)
    geracoes: int = 0
    avaliacoes: int = 0


def _reparar(inst: InstanciaQ1, genes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Garante no máximo inst.max_descontos SKUs com preço abaixo do atual."""
    niveis = genes + inst.nivel_min
    desconto = niveis < inst.idx_nivel_atual
    excesso = desconto.sum(axis=1) - inst.max_descontos
    alvo = inst.nivel_padrao() - inst.nivel_min           # gene do preço atual (ou menor viável acima)
    for p in np.flatnonzero(excesso > 0):
        escolhidos = rng.choice(np.flatnonzero(desconto[p]), size=excesso[p], replace=False)
        genes[p, escolhidos] = alvo[escolhidos]
    return genes


def _avaliar(inst: InstanciaQ1, genes: np.ndarray) -> np.ndarray:
    return fitness(inst, genes + inst.nivel_min)


def algoritmo_genetico(inst: InstanciaQ1, cfg: ConfigGA = CONFIG_A, seed: int = 0) -> ResultadoGA:
    cfg.validar()
    rng = np.random.default_rng(seed)
    n, P = inst.n, cfg.populacao
    pm = cfg.taxa_mutacao if cfg.taxa_mutacao is not None else 1.0 / n
    limite = inst.n_opcoes                                   # (n,) valores possíveis de cada gene

    pop = (rng.random((P, n)) * limite).astype(int)
    if cfg.semente_heuristica:
        pop[0] = inst.nivel_padrao() - inst.nivel_min
    pop = _reparar(inst, pop, rng)
    fit = _avaliar(inst, pop)
    avaliacoes = P

    melhor_idx = int(np.argmax(fit))
    melhor_x, melhor_f = pop[melhor_idx].copy(), float(fit[melhor_idx])
    hist_m, hist_med = [melhor_f], [float(np.mean(fit))]
    sem_melhora, g = 0, 0

    for g in range(1, cfg.geracoes_max + 1):
        # elitismo
        ordem = np.argsort(-fit)
        elite = pop[ordem[:cfg.elitismo]]

        # seleção por torneio (vetorizada): para cada filho sorteia k competidores
        n_filhos = P - cfg.elitismo
        comp = rng.integers(0, P, size=(2 * n_filhos, cfg.torneio))
        vencedores = comp[np.arange(2 * n_filhos), np.argmax(fit[comp], axis=1)]
        pais1, pais2 = pop[vencedores[:n_filhos]], pop[vencedores[n_filhos:]]

        # crossover
        if cfg.tipo_crossover == "uniforme":
            mascara = rng.random((n_filhos, n)) < 0.5
        else:
            corte = rng.integers(1, n, size=(n_filhos, 1)) if n > 1 else np.ones((n_filhos, 1), int)
            mascara = np.arange(n)[None, :] < corte
        cruza = rng.random(n_filhos) < cfg.taxa_crossover
        mascara[~cruza] = True                                 # sem crossover: filho = pai1
        filhos = np.where(mascara, pais1, pais2)

        # mutação: reinicializa gene com probabilidade pm
        muta = rng.random((n_filhos, n)) < pm
        aleatorio = (rng.random((n_filhos, n)) * limite).astype(int)
        filhos = np.where(muta, aleatorio, filhos)
        filhos = _reparar(inst, filhos, rng)

        pop = np.vstack([elite, filhos])
        fit = np.concatenate([fit[ordem[:cfg.elitismo]], _avaliar(inst, filhos)])
        avaliacoes += n_filhos

        i = int(np.argmax(fit))
        if fit[i] > melhor_f + 1e-9:
            melhor_x, melhor_f, sem_melhora = pop[i].copy(), float(fit[i]), 0
        else:
            sem_melhora += 1
        hist_m.append(melhor_f)
        hist_med.append(float(np.mean(fit[np.isfinite(fit)])))
        if sem_melhora >= cfg.paciencia:
            break

    return ResultadoGA(melhor_x + inst.nivel_min, melhor_f, hist_m, hist_med, g, avaliacoes)
