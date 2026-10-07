"""Q1 — medição de desempenho (tempo com perf_counter, memória com tracemalloc).

Gera em questao_01/resultados/:
  desempenho_escala.csv     BF, GA e Greedy em entradas pequena, média e grande
  otimizacao_antes_depois.csv  avaliação em Python puro x vetorizada em NumPy
  desempenho_escala.png

Uso: python questao_01/benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comum.dados import carregar_q1  # noqa: E402
from comum.medicao import medir  # noqa: E402
from questao_01.brute_force import brute_force, brute_force_ingenuo  # noqa: E402
from questao_01.genetic_algorithm import CONFIG_A, algoritmo_genetico  # noqa: E402
from questao_01.greedy import alocar_incentivos  # noqa: E402
from questao_01.modelo import avaliar, construir_instancia, fitness, fitness_ingenuo, subinstancia  # noqa: E402

SAIDA = Path(__file__).resolve().parent / "resultados"


def main():
    cat, cfg = carregar_q1()
    inst = construir_instancia(cat, cfg)
    seed = cfg["seed_final"]
    atual = avaliar(inst, inst.nivel_padrao())
    ordem = np.argsort(-(atual["vendas"][0] * inst.contrib[np.arange(inst.n), inst.nivel_padrao()]))
    linhas = []

    # Brute Force: n = 4..8 (crescimento exponencial L^n)
    for n in [4, 5, 6, 7, 8]:
        sub = subinstancia(inst, np.sort(ordem[:n]))
        m = medir(brute_force, sub)
        linhas.append({"algoritmo": "Brute Force", "n_skus": n, "tamanho": m.resultado.combinacoes,
                       "unidade_tamanho": "combinações", "tempo_s": m.tempo_s, "memoria_kb": m.memoria_pico_kb})

    # GA e Greedy: pequena (8), média (15), grande (22 = catálogo), muito grande (44 = catálogo duplicado)
    tamanhos = {8: np.sort(ordem[:8]), 15: np.sort(ordem[:15]), 22: np.arange(inst.n),
                44: np.r_[np.arange(inst.n), np.arange(inst.n)]}
    for n, idx in tamanhos.items():
        sub = subinstancia(inst, idx)
        m = medir(algoritmo_genetico, sub, CONFIG_A, seed)
        g = m.resultado
        linhas.append({"algoritmo": "GA (config A)", "n_skus": n, "tamanho": g.avaliacoes,
                       "unidade_tamanho": f"avaliações ({g.geracoes} gerações)", "tempo_s": m.tempo_s,
                       "memoria_kb": m.memoria_pico_kb})
        mg = medir(alocar_incentivos, sub, g.niveis, cfg["orcamento_promocional"] * n / inst.n)
        linhas.append({"algoritmo": "Greedy", "n_skus": n, "tamanho": n * 4, "unidade_tamanho": "degraus (n*J)",
                       "tempo_s": mg.tempo_s, "memoria_kb": mg.memoria_pico_kb})
    df = pd.DataFrame(linhas)
    df["tempo_s"] = df["tempo_s"].round(5)
    df["memoria_kb"] = df["memoria_kb"].round(1)
    df.to_csv(SAIDA / "desempenho_escala.csv", index=False)
    print(df.to_string(index=False))

    # Otimização antes/depois
    otim = []
    sub6 = subinstancia(inst, np.sort(ordem[:6]))
    a = medir(brute_force_ingenuo, sub6)
    b = medir(brute_force, sub6)
    assert abs(a.resultado.fitness - b.resultado.fitness) < 1e-6
    otim.append({"trecho": "Brute Force 6 SKUs (15.680 comb.)", "antes": "itertools + laço Python",
                 "depois": "lotes NumPy (unravel_index)", "tempo_antes_s": a.tempo_s, "tempo_depois_s": b.tempo_s,
                 "memoria_antes_kb": a.memoria_pico_kb, "memoria_depois_kb": b.memoria_pico_kb})
    rng = np.random.default_rng(seed)
    pop = inst.nivel_min + (rng.random((100, inst.n)) * inst.n_opcoes).astype(int)
    a = medir(lambda: [fitness_ingenuo(inst, x) for x in pop])
    b = medir(fitness, inst, pop)
    assert np.allclose(np.array(a.resultado), np.where(np.isfinite(b.resultado), b.resultado, -np.inf))
    otim.append({"trecho": "Fitness de 1 geração do GA (P=100, n=22)", "antes": "laço Python por indivíduo",
                 "depois": "matriz P x n + one-hot de categorias", "tempo_antes_s": a.tempo_s,
                 "tempo_depois_s": b.tempo_s, "memoria_antes_kb": a.memoria_pico_kb,
                 "memoria_depois_kb": b.memoria_pico_kb})
    df_o = pd.DataFrame(otim)
    df_o["speedup"] = (df_o["tempo_antes_s"] / df_o["tempo_depois_s"]).round(1)
    df_o.round(5).to_csv(SAIDA / "otimizacao_antes_depois.csv", index=False)
    print(df_o.round(5).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bf = df[df.algoritmo == "Brute Force"]
    axes[0].semilogy(bf.n_skus, bf.tempo_s, "o-", label="Brute Force (O(L^n·n))")
    axes[0].set_xlabel("nº de SKUs"); axes[0].set_ylabel("tempo (s, log)"); axes[0].grid(alpha=0.3)
    axes[0].set_title("Brute Force cresce ~7x por SKU"); axes[0].legend()
    for alg, mk in [("GA (config A)", "s-"), ("Greedy", "^-")]:
        d = df[df.algoritmo == alg]
        axes[1].plot(d.n_skus, d.tempo_s, mk, label=alg)
    axes[1].set_xlabel("nº de SKUs"); axes[1].set_ylabel("tempo (s)"); axes[1].grid(alpha=0.3)
    axes[1].set_title("GA e Greedy (polinomiais em n)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(SAIDA / "desempenho_escala.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
