"""Q2 — escalabilidade em tempo e memória (entrega 2.5.7) e otimização da DP.

Três tamanhos de entrada (n SKUs = 8, 16, 24) + estoque escalado (x2, x4) para estressar
o espaço de estados da DP, e variação do nº de cenários do Monte Carlo.

Gera em questao_02/resultados/: desempenho_escala.csv, desempenho_cenarios.csv,
otimizacao_dp.csv e desempenho_escala.png.

Uso: python questao_02/benchmark.py
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

from comum.dados import carregar_q2  # noqa: E402
from comum.medicao import medir  # noqa: E402
from questao_02.dynamic_programming import (programacao_dinamica, resolver_sku, resolver_sku_top_down,  # noqa: E402
                                            trajetoria_esperada)
from questao_02.greedy import alocar_destaques  # noqa: E402
from questao_02.modelo import construir_instancia, subinstancia  # noqa: E402
from questao_02.monte_carlo import estimar_estatisticas, gerar_cenarios, simular  # noqa: E402

SAIDA = Path(__file__).resolve().parent / "resultados"


def pipeline(inst, seed, n_c=2000, n_v=5000):
    m_mc = medir(lambda: estimar_estatisticas(inst, gerar_cenarios(inst, n_c, seed)))
    est = m_mc.resultado
    m_dp = medir(programacao_dinamica, inst, est, 0.5)
    dp = m_dp.resultado
    plano = trajetoria_esperada(inst, est, dp)
    m_gr = medir(alocar_destaques, inst, est, dp, plano)
    cv = gerar_cenarios(inst, n_v, seed + 7777)
    m_v = medir(simular, inst, cv, dp.politica, m_gr.resultado.destaques, m_gr.resultado.custo)
    return m_mc, m_dp, m_gr, m_v


def main():
    cat, cfg = carregar_q2()
    inst = construir_instancia(cat, cfg)
    seed = cfg["seed_final"]
    linhas = []
    casos = [("pequena", 8, 1), ("média", 16, 1), ("grande", 24, 1), ("grande x2 estoque", 24, 2),
             ("grande x4 estoque", 24, 4)]
    for nome, n, esc in casos:
        sub = subinstancia(inst, np.arange(n), escala_estoque=esc)
        estados = int(sum(int(s) + 1 for s in sub.estoque) * sub.T)
        for alg, m in zip(["Monte Carlo (2000)", "DP", "Greedy", "MC validação (5000)"], pipeline(sub, seed)):
            linhas.append({"entrada": nome, "n_skus": n, "estados_dp (sum T*(S+1))": estados, "algoritmo": alg,
                           "tempo_s": round(m.tempo_s, 5), "memoria_mb": round(m.memoria_pico_kb / 1024, 3)})
    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA / "desempenho_escala.csv", index=False)
    print(df.to_string(index=False))

    cen_l = []
    for N in [1000, 2000, 5000, 10000]:
        m = medir(lambda: estimar_estatisticas(inst, gerar_cenarios(inst, N, seed)))
        cen_l.append({"cenarios": N, "tempo_s": round(m.tempo_s, 4), "memoria_mb": round(m.memoria_pico_kb / 1024, 2)})
    df_c = pd.DataFrame(cen_l)
    df_c.to_csv(SAIDA / "desempenho_cenarios.csv", index=False)
    print(df_c.to_string(index=False))

    # Otimização: DP Top-Down recursiva em Python x Bottom-Up vetorizada em NumPy (mesmo valor)
    est = estimar_estatisticas(inst, gerar_cenarios(inst, 2000, seed))
    pesos = inst.pesos_urgencia(0.5)
    i = int(np.argmax(inst.estoque))
    a = medir(resolver_sku_top_down, inst, est, i, pesos)
    b = medir(resolver_sku, inst, est, i, pesos)
    assert abs(a.resultado[0] - b.resultado[0][0, inst.estoque[i]]) < 1e-6
    df_o = pd.DataFrame([{
        "trecho": f"DP do SKU {inst.skus[i]} (S={inst.estoque[i]}, T=12, K=7, B=10)",
        "antes": "Top-Down recursivo + lru_cache (Python puro)", "depois": "Bottom-Up vetorizado em s (NumPy)",
        "tempo_antes_s": round(a.tempo_s, 5), "tempo_depois_s": round(b.tempo_s, 5),
        "memoria_antes_kb": round(a.memoria_pico_kb, 1), "memoria_depois_kb": round(b.memoria_pico_kb, 1),
        "speedup": round(a.tempo_s / b.tempo_s, 1), "subproblemas_reusados(hits)": a.resultado[1].hits,
        "subproblemas_distintos(misses)": a.resultado[1].misses,
    }])
    df_o.to_csv(SAIDA / "otimizacao_dp.csv", index=False)
    print(df_o.T.to_string(header=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for alg, mk in [("Monte Carlo (2000)", "o-"), ("DP", "s-"), ("Greedy", "^-"), ("MC validação (5000)", "d-")]:
        d = df[df.algoritmo == alg]
        axes[0].plot(d["estados_dp (sum T*(S+1))"], d.tempo_s, mk, label=alg)
        axes[1].plot(d["estados_dp (sum T*(S+1))"], d.memoria_mb, mk, label=alg)
    for ax, y in zip(axes, ["tempo (s)", "memória de pico (MB)"]):
        ax.set_xlabel("estados da DP (soma de T*(S+1))"); ax.set_ylabel(y); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Q2 — escalabilidade (8, 16, 24 SKUs; 24 SKUs com estoque x2 e x4)")
    fig.tight_layout(); fig.savefig(SAIDA / "desempenho_escala.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
