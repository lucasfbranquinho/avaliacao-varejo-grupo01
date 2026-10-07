"""Questão 1 — Operação Caixa 90: Brute Force -> Algoritmo Genético -> Greedy.

Fluxo (Nesta questão utilizamos 3 algoritmos):
  1. Brute Force resolve EXATAMENTE um subconjunto pequeno (padrão: 8 SKUs).
  2. O GA roda no MESMO subconjunto e o gap% em relação ao ótimo do BF valida o GA.
  3. O GA validado roda no catálogo completo (configs A e B comparadas).
  4. A política de preços do GA alimenta o Greedy, que distribui a verba de cupons.
  5. Relatório financeiro: caixa, margem, estoque, cobertura da dívida e meta.

Exemplos:
  python questao_01/main.py
  python questao_01/main.py --seed 1002 --modo completo
  python questao_01/main.py --meta-mult 1.10 --logistica-mult 1.15 --orcamento-mult 0.75
  python questao_01/main.py --remover-sku Q1-016 --ga-pop 60 --ga-mutacao 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comum.dados import PASTA_DADOS, carregar_q1, remover_skus  # noqa: E402
from comum.medicao import medir  # noqa: E402
from questao_01.brute_force import brute_force  # noqa: E402
from questao_01.genetic_algorithm import CONFIG_A, CONFIG_B, algoritmo_genetico  # noqa: E402
from questao_01.greedy import alocacao_uniforme, alocar_incentivos  # noqa: E402
from questao_01.modelo import ParametrosQ1, avaliar, construir_instancia, subinstancia  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None, help="seed (padrão: seed_final do config)")
    ap.add_argument("--modo", choices=["rapido", "completo"], default="completo",
                    help="completo = 5 repetições do GA por config; rapido = 2")
    ap.add_argument("--dados", type=Path, default=PASTA_DADOS)
    ap.add_argument("--grupo", default="G01")
    ap.add_argument("--saida", type=Path, default=Path(__file__).resolve().parent / "resultados")
    ap.add_argument("--subconjunto", type=int, default=8, help="nº de SKUs do Brute Force (>= 6)")
    # cenário-surpresa
    ap.add_argument("--meta-mult", type=float, default=1.0)
    ap.add_argument("--logistica-mult", type=float, default=1.0)
    ap.add_argument("--orcamento-mult", type=float, default=1.0)
    ap.add_argument("--margem-minima", type=float, default=None, help="sobrescreve margem mínima")
    ap.add_argument("--remover-sku", nargs="*", default=None)
    ap.add_argument("--max-frac-desconto", type=float, default=0.40)
    # GA
    ap.add_argument("--ga-pop", type=int, default=None, help="população da config A")
    ap.add_argument("--ga-mutacao", type=float, default=None, help="taxa de mutação da config A")
    ap.add_argument("--ga-geracoes", type=int, default=None)
    return ap.parse_args(argv)


def executar(args) -> dict:
    cat, cfg = carregar_q1(args.dados, args.grupo)
    cat = remover_skus(cat, args.remover_sku)
    if args.margem_minima is not None:
        cfg = {**cfg, "margem_minima_percentual": args.margem_minima}
    seed = args.seed if args.seed is not None else cfg["seed_final"]
    params = ParametrosQ1(meta_mult=args.meta_mult, logistica_mult=args.logistica_mult,
                          orcamento_mult=args.orcamento_mult, max_frac_desconto=args.max_frac_desconto)
    inst = construir_instancia(cat, cfg, params)
    meta = cfg["meta_caixa_90d"] * args.meta_mult
    orcamento = cfg["orcamento_promocional"] * args.orcamento_mult
    divida = cfg["divida_total_90d"]
    args.saida.mkdir(parents=True, exist_ok=True)
    repeticoes = 5 if args.modo == "completo" else 2

    cfg_a = CONFIG_A
    if args.ga_pop is not None:
        cfg_a = replace(cfg_a, populacao=args.ga_pop)
    if args.ga_mutacao is not None:
        cfg_a = replace(cfg_a, taxa_mutacao=args.ga_mutacao)
    if args.ga_geracoes is not None:
        cfg_a = replace(cfg_a, geracoes_max=args.ga_geracoes)
    if cfg_a is not CONFIG_A:
        cfg_a = replace(cfg_a, nome=f"A* (P={cfg_a.populacao}, pm={cfg_a.taxa_mutacao or '1/n'}, "
                                    f"G={cfg_a.geracoes_max})")
    configs = [cfg_a, CONFIG_B]

    print(f"Instância: {inst.n} SKUs ativos | suspensos (sem nível que respeite a margem mínima): "
          f"{inst.suspensos} | seed={seed}")

    # ---- política atual (linha de base) -------------------------------------------------
    atual = avaliar(inst, inst.nivel_padrao())

    # ---- Etapa A: Brute Force no subconjunto -------------------------------------------
    caixa_sku = atual["vendas"][0] * inst.contrib[np.arange(inst.n), inst.nivel_padrao()]
    n_sub = min(max(6, args.subconjunto), inst.n)
    idx_sub = np.sort(np.argsort(-caixa_sku)[:n_sub])        # SKUs de maior peso no caixa
    sub = subinstancia(inst, idx_sub)
    m_bf = medir(brute_force, sub)
    bf = m_bf.resultado
    print(f"\n[A] Brute Force em {n_sub} SKUs: {bf.combinacoes:,} combinações "
          f"({bf.viaveis:,} viáveis) | ótimo = R$ {bf.fitness:,.2f} | {m_bf.tempo_s:.3f}s | "
          f"{m_bf.memoria_pico_kb:,.0f} KB")

    # ---- Etapa B1: GA no mesmo subconjunto -> gap -------------------------------------
    linhas_bf = []
    for c in configs:
        for r in range(repeticoes):
            m = medir(algoritmo_genetico, sub, c, seed + r)
            g = m.resultado
            linhas_bf.append({
                "config": c.nome, "seed": seed + r, "fitness_ga": round(g.fitness, 2),
                "fitness_bf": round(bf.fitness, 2), "gap_pct": round(100 * (bf.fitness - g.fitness) / abs(bf.fitness), 4),
                "geracoes": g.geracoes, "avaliacoes": g.avaliacoes,
                "tempo_ga_s": round(m.tempo_s, 4), "tempo_bf_s": round(m_bf.tempo_s, 4),
                "memoria_ga_kb": round(m.memoria_pico_kb, 1), "memoria_bf_kb": round(m_bf.memoria_pico_kb, 1),
                "politica_ga": " ".join(f"{v:.2f}" for v in sub.niveis[g.niveis]),
                "politica_bf": " ".join(f"{v:.2f}" for v in sub.niveis[bf.niveis]),
            })
    df_bf = pd.DataFrame(linhas_bf)
    df_bf.to_csv(args.saida / "bf_vs_ga_subconjunto.csv", index=False)
    print("[B1] GA x BF no subconjunto (gap% médio por config):")
    print(df_bf.groupby("config")[["gap_pct", "geracoes", "tempo_ga_s"]].mean().round(4).to_string())

    # ---- Etapa B2: GA no catálogo completo --------------------------------------------
    linhas_ga, historicos, melhor = [], {}, None
    for c in configs:
        for r in range(repeticoes):
            m = medir(algoritmo_genetico, inst, c, seed + r)
            g = m.resultado
            linhas_ga.append({"config": c.nome, "seed": seed + r, "fitness": round(g.fitness, 2),
                              "geracoes": g.geracoes, "avaliacoes": g.avaliacoes,
                              "tempo_s": round(m.tempo_s, 4), "memoria_kb": round(m.memoria_pico_kb, 1)})
            historicos.setdefault(c.nome, []).append(g)
            if melhor is None or g.fitness > melhor[1].fitness:
                melhor = (c.nome, g)
    df_ga = pd.DataFrame(linhas_ga)
    df_ga.to_csv(args.saida / "ga_configuracoes.csv", index=False)
    resumo_ga = df_ga.groupby("config").agg(fitness_medio=("fitness", "mean"), fitness_desvio=("fitness", "std"),
                                            fitness_max=("fitness", "max"), geracoes=("geracoes", "mean"),
                                            tempo_s=("tempo_s", "mean"), memoria_kb=("memoria_kb", "mean")).round(2)
    resumo_ga.to_csv(args.saida / "ga_configuracoes_resumo.csv")
    print("\n[B2] GA no catálogo completo:")
    print(resumo_ga.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    cores = {configs[0].nome: "#1f77b4", configs[1].nome: "#d62728"}
    for nome, gs in historicos.items():
        for k, g in enumerate(gs):
            ax.plot(g.historico_melhor, color=cores[nome], alpha=0.9 if k == 0 else 0.35,
                    label=f"{nome} — melhor" if k == 0 else None)
        ax.plot(gs[0].historico_media, color=cores[nome], ls="--", alpha=0.6, label=f"{nome} — média (seed {seed})")
    fit_atual = float(atual["fitness"][0])
    ax.axhline(fit_atual, color="gray", ls=":", label="política atual (preço 1.00)")
    ax.set_xlabel("geração"); ax.set_ylabel("fitness (R$)"); ax.set_title(f"Q1 — convergência do GA ({inst.n} SKUs)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    vals = [v for gs in historicos.values() for g in gs for v in g.historico_melhor]
    ax.set_ylim(min(np.percentile(vals, 1), fit_atual) * 0.98, max(vals) * 1.01)
    fig.tight_layout(); fig.savefig(args.saida / "ga_fitness_por_geracao.png", dpi=130); plt.close(fig)

    # ---- Etapa C: Greedy sobre a política do GA ---------------------------------------
    nome_melhor, g_melhor = melhor
    m_gr = medir(alocar_incentivos, inst, g_melhor.niveis, orcamento)
    gr = m_gr.resultado
    gr.tabela.to_csv(args.saida / "greedy_antes_depois.csv", index=False)
    taxa_u, caixa_u, gasto_u = alocacao_uniforme(inst, g_melhor.niveis, orcamento)
    print(f"\n[C] Greedy (verba R$ {orcamento:,.2f}): caixa {gr.caixa_antes:,.2f} -> {gr.caixa_depois:,.2f} "
          f"| gasto R$ {gr.gasto.sum():,.2f} | {m_gr.tempo_s*1000:.2f} ms")
    print(f"    linha de base cupom uniforme {taxa_u:.0%}: caixa {caixa_u:,.2f} gasto {gasto_u:,.2f}")

    # ---- Relatório financeiro ----------------------------------------------------------
    lin = np.arange(inst.n)
    preco = inst.preco[lin, g_melhor.niveis]
    receita = float((gr.vendas * preco).sum())
    estoque_ini_total = float(cat["estoque"].sum())
    estoque_susp = float(cat.loc[cat["sku"].isin(inst.suspensos), "estoque"].sum())
    resid = inst.estoque - gr.vendas
    resid_total = float(resid.sum()) + estoque_susp
    valor_resid = float((resid * inst.custo).sum() + (cat.loc[cat["sku"].isin(inst.suspensos), "estoque"]
                                                      * cat.loc[cat["sku"].isin(inst.suspensos), "custo_unitario"]).sum())
    resultado = {
        "grupo": cfg["grupo"], "seed": seed, "skus_ativos": inst.n, "skus_suspensos": inst.suspensos,
        "cenario": {"meta_mult": args.meta_mult, "logistica_mult": args.logistica_mult,
                    "orcamento_mult": args.orcamento_mult, "remover_sku": args.remover_sku or [],
                    "margem_minima": cfg["margem_minima_percentual"], "ga_config_A": cfg_a.nome},
        "brute_force": {"n_skus": n_sub, "skus": sub.skus, "combinacoes": bf.combinacoes,
                        "otimo": round(bf.fitness, 2), "tempo_s": round(m_bf.tempo_s, 4),
                        "gap_medio_pct": df_bf.groupby("config")["gap_pct"].mean().round(4).to_dict()},
        "ga_melhor_config": nome_melhor, "ga_fitness": round(g_melhor.fitness, 2),
        "caixa_politica_atual": round(float(atual["caixa"][0]), 2),
        "caixa_apos_ga": round(gr.caixa_antes, 2),
        "caixa_final_apos_greedy": round(gr.caixa_depois, 2),
        "receita_bruta": round(receita, 2),
        "margem_contribuicao_pct": round(100 * gr.caixa_depois / receita, 2),
        "meta_caixa": round(meta, 2),
        "meta_atingida": bool(gr.caixa_depois >= meta),
        "folga_meta": round(gr.caixa_depois - meta, 2),
        "divida_90d": divida,
        "cobertura_divida_pct": round(100 * gr.caixa_depois / divida, 2),
        "orcamento_promocional": round(orcamento, 2),
        "orcamento_usado": round(float(gr.gasto.sum()), 2),
        "orcamento_usado_pct": round(100 * float(gr.gasto.sum()) / orcamento, 2) if orcamento else 0.0,
        "retorno_cupom_por_real": round((gr.caixa_depois - gr.caixa_antes) / max(gr.gasto.sum(), 1e-9), 3),
        "caixa_liquido_se_verba_propria": round(gr.caixa_depois - float(gr.gasto.sum()), 2),
        "estoque_inicial_unid": int(estoque_ini_total),
        "estoque_remanescente_unid": int(resid_total),
        "estoque_remanescente_pct": round(100 * resid_total / estoque_ini_total, 2),
        "estoque_remanescente_valor_custo": round(valor_resid, 2),
        "skus_com_desconto": int((inst.niveis[g_melhor.niveis] < 1).sum()),
        "limite_descontos": inst.max_descontos,
        "tempos_s": {"brute_force": round(m_bf.tempo_s, 4), "greedy": round(m_gr.tempo_s, 6),
                     "ga_medio": df_ga.groupby("config")["tempo_s"].mean().round(4).to_dict()},
    }
    pd.DataFrame({"sku": inst.skus, "categoria": inst.categorias,
                  "multiplicador": inst.niveis[g_melhor.niveis], "preco_final": preco.round(2),
                  "cupom_pct": 100 * gr.cupom, "vendas_90d": gr.vendas.astype(int),
                  "estoque_restante": resid.astype(int)}).to_csv(args.saida / "politica_final.csv", index=False)
    (args.saida / "resumo.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=========== RESULTADO FINAL — Q1 ===========")
    print(f"Caixa (política atual)        R$ {resultado['caixa_politica_atual']:>14,.2f}")
    print(f"Caixa após GA                 R$ {resultado['caixa_apos_ga']:>14,.2f}")
    print(f"Caixa após Greedy (final)     R$ {resultado['caixa_final_apos_greedy']:>14,.2f}")
    print(f"Meta de caixa 90d             R$ {meta:>14,.2f} -> "
          f"{'ATINGIDA' if resultado['meta_atingida'] else 'NÃO atingida'} (folga R$ {resultado['folga_meta']:,.2f})")
    print(f"Cobertura da dívida 90d       {resultado['cobertura_divida_pct']:.1f}% de R$ {divida:,.2f}")
    print(f"Margem de contribuição        {resultado['margem_contribuicao_pct']:.1f}%")
    print(f"Estoque remanescente          {resultado['estoque_remanescente_unid']} unid "
          f"({resultado['estoque_remanescente_pct']:.1f}%) | R$ {valor_resid:,.2f} a custo")
    print(f"Verba promocional usada       R$ {resultado['orcamento_usado']:,.2f} "
          f"({resultado['orcamento_usado_pct']:.1f}%)")
    print(f"Resultados salvos em {args.saida}")
    return resultado


if __name__ == "__main__":
    executar(parse_args())
