"""Desenha docs/arquitetura.png (fluxo de dados entre os algoritmos das duas questões).

Uso: python docs/gerar_arquitetura.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SAIDA = Path(__file__).resolve().parent / "arquitetura.png"


def caixa(ax, x, y, w, h, titulo, texto, cor):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=cor, ec="#333333", lw=1.2))
    ax.text(x + w / 2, y + h - 0.17, titulo, ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.52, texto, ha="center", va="top", fontsize=8.2, linespacing=1.35)


def seta(ax, a, b, rotulo="", curva=0.0, dy=0.12):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.5, color="#333333",
                                 connectionstyle=f"arc3,rad={curva}"))
    if rotulo:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, rotulo, ha="center", fontsize=7.8,
                style="italic", color="#444444")


fig, ax = plt.subplots(figsize=(15, 9.2))
ax.set_xlim(0, 15); ax.set_ylim(0, 9.2); ax.axis("off")

# Dados
caixa(ax, 0.2, 7.3, 3.2, 1.6, "Notebook gerador",
      "Bloco 02: G01 + RM562262\nseed_final = 1002\nCSV/JSON + manifesto SHA-256", "#eeeeee")
caixa(ax, 0.2, 4.9, 3.2, 1.7, "comum/dados.py",
      "verifica hashes do manifesto\nvalida colunas\naplica cenário-surpresa\n(parâmetros CLI, sem editar data/)", "#eeeeee")
seta(ax, (1.8, 7.3), (1.8, 6.6), "data/*.csv, *.json", dy=0.0)

# Q1
ax.text(4.0, 8.95, "QUESTÃO 1 — Operação Caixa 90", fontsize=12, weight="bold", color="#1f4e79")
caixa(ax, 4.0, 6.9, 3.2, 1.75, "A. Brute Force",
      "subconjunto de 8 SKUs\n548.800 combinações\nótimo exato (benchmark)", "#dbe9f6")
caixa(ax, 7.8, 6.9, 3.2, 1.75, "B. Algoritmo Genético",
      "1) mesmo subconjunto → gap%\n2) catálogo completo (22 SKUs)\nconfigs A x B", "#dbe9f6")
caixa(ax, 11.6, 6.9, 3.2, 1.75, "C. Greedy",
      "verba de cupons por\nbenefício marginal / R$\n(heap de degraus)", "#dbe9f6")
seta(ax, (7.2, 7.8), (7.8, 7.8), "ótimo p/ validar", dy=0.4)
seta(ax, (11.0, 7.8), (11.6, 7.8), "política de preços", dy=0.4)
caixa(ax, 7.8, 5.25, 7.0, 1.2, "Relatório Q1 (questao_01/resultados)",
      "caixa x meta, cobertura da dívida, margem, estoque remanescente, uso da verba", "#f3f8fc")
seta(ax, (13.2, 6.9), (13.2, 6.45))
seta(ax, (3.4, 5.9), (4.0, 7.3), "", curva=-0.2)

# Q2
ax.text(4.0, 4.45, "QUESTÃO 2 — Plano de Sobrevivência em 12 Semanas", fontsize=12, weight="bold", color="#1e5b2e")
caixa(ax, 4.0, 2.25, 2.6, 1.9, "A. Monte Carlo",
      "2000 cenários (seed s)\ndemanda, concorrente,\nlogística → faixas de\nquantis por semana/nível", "#dff0dc")
caixa(ax, 7.2, 2.25, 2.6, 1.9, "B. Prog. Dinâmica",
      "estado (semana, estoque)\nBellman bottom-up\nα de urgência calibrado\nno MC de construção", "#dff0dc")
caixa(ax, 10.4, 2.25, 2.0, 1.9, "C. Greedy",
      "destaques por\nsemana usando\nV(t,s) da DP", "#dff0dc")
caixa(ax, 12.9, 2.25, 1.9, 1.9, "D. Monte Carlo",
      "validação\n5000 cenários\nNOVOS\n(seed s+7777)", "#dff0dc")
seta(ax, (6.6, 3.2), (7.2, 3.2), "distribuições", dy=0.4)
seta(ax, (9.8, 3.2), (10.4, 3.2), "plano + V", dy=0.4)
seta(ax, (12.4, 3.2), (12.9, 3.2), "plano", dy=0.4)
seta(ax, (8.5, 2.25), (5.3, 2.25), "α → simula no MC de construção", curva=-0.35, dy=-0.85)
caixa(ax, 4.0, 0.2, 10.8, 1.15, "Relatório Q2 (questao_02/resultados)",
      "P(pagar semanas 4, 8, 12), caixa final médio e P5, CVaR, margem, estoque final, falhas, estresse da dívida",
      "#f3fbf2")
seta(ax, (13.85, 2.25), (13.85, 1.35))
seta(ax, (3.4, 5.2), (4.0, 3.6), "", curva=0.2)

fig.savefig(SAIDA, dpi=140, bbox_inches="tight")
print(f"salvo em {SAIDA}")
