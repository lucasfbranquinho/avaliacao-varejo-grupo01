"""Questão 2 — Plano de Sobrevivência em 12 Semanas.

Fluxo (Nesta questão utilizamos 3 algoritmos):
  A. Monte Carlo (construção, 2000 cenários): distribuição da demanda/caixa por SKU x semana x nível.
  B. Programação Dinâmica: com essas distribuições, escolhe o nível de preço ótimo para cada
     (semana, estoque restante). O peso de urgência alpha é calibrado no próprio MC de construção.
  C. Greedy: com o plano da DP (estoque e preço esperados por semana), escolhe quais SKUs
     recebem os destaques promocionais de cada semana.
  D. Monte Carlo (validação, 5000 cenários NOVOS): simula o plano completo e mede a
     probabilidade de pagar as dívidas das semanas 4, 8 e 12.

Exemplos:
  python questao_02/main.py
  python questao_02/main.py --seed 1002 --cenarios 5000
  python questao_02/main.py --concorrente-mult 0.92 --logistica-mult 1.15 --orcamento-mult 0.75
  python questao_02/main.py --dividas-mult 1.10 --remover-sku Q2-009
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comum.dados import PASTA_DADOS, carregar_q2, remover_skus  # noqa: E402
from comum.medicao import medir  # noqa: E402
from questao_02.dynamic_programming import programacao_dinamica, trajetoria_esperada  # noqa: E402
from questao_02.greedy import alocar_destaques  # noqa: E402
from questao_02.modelo import ParametrosQ2, construir_instancia  # noqa: E402
from questao_02.monte_carlo import (SEED_OFFSET_VALIDACAO, estimar_estatisticas, gerar_cenarios,  # noqa: E402
                                    politica_fixa, simular)

PERCENTIL_RISCO = 5   # percentil escolhido: P5 do caixa final (VaR 95%)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None, help="seed (padrão: seed_final do config)")
    ap.add_argument("--cenarios", type=int, default=5000, help="cenários da validação final (>= 5000)")
    ap.add_argument("--cenarios-construcao", type=int, default=None,
                    help="cenários da estimativa (padrão: n_simulacoes_recomendadas = 2000)")
    ap.add_argument("--dados", type=Path, default=PASTA_DADOS)
    ap.add_argument("--grupo", default="G01")
    ap.add_argument("--saida", type=Path, default=Path(__file__).resolve().parent / "resultados")
    # cenário-surpresa
    ap.add_argument("--concorrente-mult", type=float, default=1.0, help="ex.: 0.92 = concorrente -8%%")
    ap.add_argument("--logistica-mult", type=float, default=1.0)
    ap.add_argument("--orcamento-mult", type=float, default=1.0)
    ap.add_argument("--dividas-mult", type=float, default=1.0)
    ap.add_argument("--demanda-mult", type=float, default=1.0)
    ap.add_argument("--remover-sku", nargs="*", default=None)
    ap.add_argument("--uplift-destaque", type=float, default=0.25)
    ap.add_argument("--max-destaques", type=int, default=6)
    return ap.parse_args(argv)


def metricas(inst, r, rotulo: str) -> dict:
    cf = r.caixa_final
    p = np.percentile(cf, PERCENTIL_RISCO)
    return {
        "politica": rotulo,
        **{f"P(pagar sem {s})": round(float(v), 4) for s, v in zip(inst.semanas_divida, r.prob_pagamento)},
        "P(pagar as 3)": round(r.prob_pagar_tudo, 4),
        "risco_falha": round(1 - r.prob_pagar_tudo, 4),
        "caixa_final_medio": round(float(cf.mean()), 2),
        f"caixa_final_P{PERCENTIL_RISCO}": round(float(p), 2),
        f"CVaR{PERCENTIL_RISCO}": round(float(cf[cf <= p].mean()), 2),
        "caixa_gerado_medio": round(float(r.contribuicao.mean()), 2),
        "receita_media": round(float(r.receita.mean()), 2),
        "margem_contrib_pct": round(100 * float(r.contribuicao.mean() / r.receita.mean()), 2),
        "estoque_final_unid": round(float(r.estoque_final.sum(axis=1).mean()), 1),
        "estoque_final_valor_custo": round(float((r.estoque_final * inst.custo).sum(axis=1).mean()), 2),
        "gasto_destaques": round(r.gasto_destaques, 2),
    }


def executar(args) -> dict:
    cat, cfg = carregar_q2(args.dados, args.grupo)
    cat = remover_skus(cat, args.remover_sku)
    seed = args.seed if args.seed is not None else cfg["seed_final"]
    n_constr = args.cenarios_construcao or cfg["n_simulacoes_recomendadas"]
    if args.cenarios < 5000:
        print(f"AVISO: {args.cenarios} cenários de validação (< 5000 exigidos pelo enunciado)")
    params = ParametrosQ2(concorrente_mult=args.concorrente_mult, logistica_mult=args.logistica_mult,
                          orcamento_mult=args.orcamento_mult, dividas_mult=args.dividas_mult,
                          demanda_mult=args.demanda_mult, uplift_destaque=args.uplift_destaque,
                          max_destaques_semana=args.max_destaques)
    inst = construir_instancia(cat, cfg, params)
    args.saida.mkdir(parents=True, exist_ok=True)
    print(f"Instância: {inst.n} SKUs, {inst.T} semanas, dívidas {dict(zip(inst.semanas_divida.tolist(), inst.dividas.round(2).tolist()))}, "
          f"caixa inicial R$ {inst.caixa_inicial:,.2f}, seed={seed}")

    # ---- A. Monte Carlo de construção --------------------------------------------------
    def construir():
        cen = gerar_cenarios(inst, n_constr, seed)
        return cen, estimar_estatisticas(inst, cen)
    m_mc = medir(construir)
    cen_c, est = m_mc.resultado
    est.resumo.to_csv(args.saida / "mc_estatisticas_por_nivel.csv", index=False)
    print(f"\n[A] Monte Carlo (construção): {n_constr} cenários x {inst.T} semanas x {inst.n} SKUs x "
          f"{inst.K} níveis | {m_mc.tempo_s:.3f}s | {m_mc.memoria_pico_kb/1024:,.1f} MB")

    # ---- B. DP + calibração do alpha no MC de construção -------------------------------
    calib = []
    melhor = None
    for a in inst.params.alphas:
        dp = programacao_dinamica(inst, est, a)
        plano = trajetoria_esperada(inst, est, dp)
        gr = alocar_destaques(inst, est, dp, plano)
        r = simular(inst, cen_c, dp.politica, gr.destaques, gr.custo)
        chave = (r.prob_pagar_tudo, np.percentile(r.caixa_final, PERCENTIL_RISCO))
        calib.append({"alpha": a, "P(pagar as 3)": r.prob_pagar_tudo,
                      f"caixa_final_P{PERCENTIL_RISCO}": round(chave[1], 2),
                      "caixa_final_medio": round(float(r.caixa_final.mean()), 2)})
        if melhor is None or chave > melhor[0]:
            melhor = (chave, a)
    alpha = melhor[1]
    pd.DataFrame(calib).to_csv(args.saida / "calibracao_alpha.csv", index=False)
    m_dp = medir(programacao_dinamica, inst, est, alpha)
    dp = m_dp.resultado
    plano = trajetoria_esperada(inst, est, dp)
    plano.to_csv(args.saida / "dp_trajetoria_precos_estoque.csv", index=False)
    print(f"[B] DP: alpha* = {alpha} (calibrado no MC de construção) | {m_dp.tempo_s:.3f}s | "
          f"{m_dp.memoria_pico_kb/1024:,.1f} MB")
    print(pd.DataFrame(calib).to_string(index=False))

    # ---- C. Greedy semanal ------------------------------------------------------------
    m_gr = medir(alocar_destaques, inst, est, dp, plano)
    gr = m_gr.resultado
    gr.escolhas.to_csv(args.saida / "greedy_destaques.csv", index=False)
    print(f"[C] Greedy: {int(gr.destaques.sum())} destaques em {inst.T} semanas | gasto planejado "
          f"R$ {gr.gasto_semanal.sum():,.2f} (orçamento R$ {inst.orcamento_semanal * inst.T:,.2f}) | "
          f"{m_gr.tempo_s*1000:.1f} ms")

    # ---- D. Monte Carlo de validação (cenários novos) ---------------------------------
    seed_v = seed + SEED_OFFSET_VALIDACAO
    cen_v = gerar_cenarios(inst, args.cenarios, seed_v)
    idx_atual = int(np.argmin(np.abs(inst.niveis - 1.0)))
    m_val = medir(simular, inst, cen_v, dp.politica, gr.destaques, gr.custo)
    r_final = m_val.resultado
    r_fixo = simular(inst, cen_v, politica_fixa(inst, idx_atual))
    r_dp = simular(inst, cen_v, dp.politica)
    tab = pd.DataFrame([metricas(inst, r_fixo, "preço fixo 1.00"), metricas(inst, r_dp, "DP sem destaques"),
                        metricas(inst, r_final, "DP + Greedy (plano final)")])
    tab.to_csv(args.saida / "validacao_comparativo.csv", index=False)
    print(f"\n[D] Validação: {args.cenarios} cenários novos (seed {seed_v}) | {m_val.tempo_s:.3f}s")
    print(tab.T.to_string(header=False))

    # ---- Estresse de dívida (reaproveita a simulação: só muda o valor devido) ---------
    def prob_com_mult(r, mult):
        antes = r.caixa_antes_pagamento.copy()
        pagos_ant = np.concatenate([[0.0], np.cumsum(inst.dividas)[:-1]])
        antes = antes - (mult - 1) * pagos_ant          # dívidas anteriores maiores
        pago = antes >= mult * inst.dividas
        return pago.mean(axis=0), pago.all(axis=1).mean(), pago
    estresse = []
    for mult in np.round(np.arange(1.0, 1.81, 0.05), 2):
        linha = {"dividas_mult": mult}
        for nome, r in [("fixo", r_fixo), ("dp", r_dp), ("dp_greedy", r_final)]:
            linha[f"P_tudo_{nome}"] = round(float(prob_com_mult(r, mult)[1]), 4)
        estresse.append(linha)
    df_est = pd.DataFrame(estresse)
    df_est.to_csv(args.saida / "estresse_dividas.csv", index=False)
    grade = np.arange(1.0, 2.0, 0.005)
    suport = {}
    for nome, r in [("preço fixo", r_fixo), ("DP", r_dp), ("DP + Greedy", r_final)]:
        ok = [m for m in grade if prob_com_mult(r, m)[1] >= 1 - inst.limite_risco]
        suport[nome] = round(float(max(ok)), 3) if ok else None     # None: risco já excede o limite em x1.0
    print("\nEstresse: multiplicador máximo da dívida (sobre a dívida do cenário) com risco <= "
          f"{inst.limite_risco:.0%}: " + ", ".join(f"{k}: " + (f"x{v:.3f}" if v else "nenhum (< x1.0)")
                                                    for k, v in suport.items()))

    # cenários de falha: no plano base e no primeiro nível de estresse em que aparecem
    falhas_base = int((~r_final.pago.all(axis=1)).sum())
    m_falha = next((e["dividas_mult"] for e in estresse if e["P_tudo_dp_greedy"] < 0.99), None)
    analise_falha = {"falhas_no_cenario_base": falhas_base, "n_cenarios": args.cenarios}
    if m_falha is not None:
        _, _, pago_m = prob_com_mult(r_final, m_falha)
        falha = ~pago_m.all(axis=1)
        choque_med = cen_v.choque_dem.mean(axis=(1, 2))
        log_med = (cen_v.logistica / inst.logistica).mean(axis=(1, 2))
        pc_med = (cen_v.pc / inst.pc0).mean(axis=(1, 2))
        analise_falha.update({
            "dividas_mult_analisado": float(m_falha), "cenarios_com_falha": int(falha.sum()),
            "semana_da_primeira_falha": {int(s): int(((~pago_m).argmax(axis=1)[falha] == j).sum())
                                         for j, s in enumerate(inst.semanas_divida)},
            "choque_demanda_medio_falha": round(float(choque_med[falha].mean()), 4),
            "choque_demanda_medio_sucesso": round(float(choque_med[~falha].mean()), 4),
            "logistica_rel_media_falha": round(float(log_med[falha].mean()), 4),
            "logistica_rel_media_sucesso": round(float(log_med[~falha].mean()), 4),
            "concorrente_rel_medio_falha": round(float(pc_med[falha].mean()), 4),
            "concorrente_rel_medio_sucesso": round(float(pc_med[~falha].mean()), 4),
        })

    # ---- Gráficos ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for r, nome, cor in [(r_fixo, "preço fixo 1.00", "#7f7f7f"), (r_dp, "DP sem destaques", "#1f77b4"),
                         (r_final, "DP + Greedy", "#2ca02c")]:
        ax.hist(r.caixa_final / 1000, bins=60, alpha=0.55, color=cor, label=nome)
    ax.axvline(0, color="red", ls="--", label="caixa zero após as dívidas")
    ax.axvline(np.percentile(r_final.caixa_final, PERCENTIL_RISCO) / 1000, color="#2ca02c", ls=":",
               label=f"P{PERCENTIL_RISCO} do plano final")
    ax.set_xlabel("caixa final após pagar as 3 dívidas (R$ mil)"); ax.set_ylabel("nº de cenários")
    ax.set_title(f"Q2 — distribuição do caixa final ({args.cenarios} cenários de validação)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(args.saida / "distribuicao_caixa_final.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    semanas = np.arange(1, inst.T + 1)
    for r, nome, cor in [(r_fixo, "preço fixo", "#7f7f7f"), (r_final, "DP + Greedy", "#2ca02c")]:
        pct = np.percentile(r.caixa_semanal, [5, 50, 95], axis=0) / 1000
        ax.fill_between(semanas, pct[0], pct[2], color=cor, alpha=0.2)
        ax.plot(semanas, pct[1], color=cor, label=f"{nome} (mediana e faixa P5–P95)")
    for s, d in zip(inst.semanas_divida, inst.dividas):
        ax.annotate(f"paga R$ {d/1000:.0f} mil", (s, 0), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="red")
    ax.axhline(0, color="red", lw=0.8)
    ax.set_xlabel("semana"); ax.set_ylabel("caixa após pagamentos (R$ mil)"); ax.set_xticks(semanas)
    ax.set_title("Q2 — evolução do caixa"); ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(args.saida / "evolucao_caixa.png", dpi=130); plt.close(fig)

    # trajetória de preço e estoque de produtos representativos (maior caixa esperado)
    vendas_sku = (r_final.estoque_semana[:, 0] - r_final.estoque_final).mean(axis=0)
    rep = list(np.argsort(-vendas_sku * inst.receita_unit[:, idx_atual])[:4])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, i in zip(axes.ravel(), rep):
        est_p = np.percentile(r_final.estoque_semana[:, :, i], [5, 50, 95], axis=0)
        t13 = np.arange(0, inst.T + 1)
        ax.fill_between(t13, est_p[0], est_p[2], color="#ff7f0e", alpha=0.2)
        ax.plot(t13, est_p[1], color="#ff7f0e", label="estoque (mediana, P5–P95)")
        pl = plano[plano["sku"] == inst.skus[i]]
        ax.plot(np.r_[0, pl["semana"]], np.r_[pl["estoque_inicio"].iloc[0], pl["estoque_fim"]], color="#ff7f0e",
                ls=":", label="plano DP (venda esperada, sem destaques)")
        ax2 = ax.twinx()
        mult = pl["multiplicador"].to_numpy()
        ax2.step(np.r_[0, pl["semana"]], np.r_[mult, mult[-1]], where="post", color="#1f77b4")
        ax2.set_ylim(0.78, 1.16); ax2.set_ylabel("multiplicador", color="#1f77b4")
        dest = gr.destaques[:, i]
        ax.scatter(np.flatnonzero(dest), np.zeros(dest.sum()), marker="^", color="#2ca02c",
                   label="destaque (Greedy)", zorder=5)
        ax.set_title(f"{inst.skus[i]} ({inst.categorias[i]})", fontsize=10)
        ax.set_ylabel("estoque (unid)"); ax.grid(alpha=0.3)
    axes[0, 0].plot([], [], color="#1f77b4", label="preço do plano (x preço atual)")
    axes[0, 0].legend(fontsize=7, loc="lower left"); axes[1, 0].set_xlabel("semana"); axes[1, 1].set_xlabel("semana")
    fig.suptitle("Q2 — trajetória semanal de preço e estoque (produtos representativos)")
    fig.tight_layout(); fig.savefig(args.saida / "trajetorias_representativas.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df_est["dividas_mult"], df_est["P_tudo_fixo"], "o-", color="#7f7f7f", label="preço fixo 1.00")
    ax.plot(df_est["dividas_mult"], df_est["P_tudo_dp"], "s-", color="#1f77b4", label="DP sem destaques")
    ax.plot(df_est["dividas_mult"], df_est["P_tudo_dp_greedy"], "^-", color="#2ca02c", label="DP + Greedy")
    ax.axhline(1 - inst.limite_risco, color="red", ls="--", label=f"limite de risco ({inst.limite_risco:.0%})")
    ax.set_xlabel("multiplicador das dívidas"); ax.set_ylabel("P(pagar as 3 dívidas)")
    ax.set_title("Q2 — teste de estresse da dívida"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.saida / "estresse_dividas.png", dpi=130); plt.close(fig)

    # ---- Resumo -------------------------------------------------------------------------
    final = tab.iloc[2].to_dict()
    resultado = {
        "grupo": cfg["grupo"], "seed_construcao": seed, "seed_validacao": seed_v,
        "cenarios_construcao": n_constr, "cenarios_validacao": args.cenarios,
        "cenario": {"concorrente_mult": args.concorrente_mult, "logistica_mult": args.logistica_mult,
                    "orcamento_mult": args.orcamento_mult, "dividas_mult": args.dividas_mult,
                    "demanda_mult": args.demanda_mult, "remover_sku": args.remover_sku or []},
        "alpha_calibrado": alpha, "plano_final": final,
        "risco_dentro_do_limite": bool(final["risco_falha"] <= inst.limite_risco),
        "skus_pausados_todo_horizonte": [inst.skus[i] for i in range(inst.n)
                                         if (plano[plano["sku"] == inst.skus[i]]["acao"] == inst.PAUSAR).all()],
        "destaques_total": int(gr.destaques.sum()),
        "multiplicador_divida_suportavel": suport,
        "analise_falhas": analise_falha,
        "tempos_s": {"monte_carlo_construcao": round(m_mc.tempo_s, 4), "dp": round(m_dp.tempo_s, 4),
                     "greedy": round(m_gr.tempo_s, 4), "monte_carlo_validacao": round(m_val.tempo_s, 4)},
        "memoria_pico_mb": {"monte_carlo_construcao": round(m_mc.memoria_pico_kb / 1024, 2),
                            "dp": round(m_dp.memoria_pico_kb / 1024, 2),
                            "greedy": round(m_gr.memoria_pico_kb / 1024, 2),
                            "monte_carlo_validacao": round(m_val.memoria_pico_kb / 1024, 2)},
    }
    (args.saida / "resumo.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2, default=float),
                                           encoding="utf-8")
    print("\n=========== RESULTADO FINAL — Q2 (plano DP + Greedy, validação) ===========")
    for s, v in zip(inst.semanas_divida, r_final.prob_pagamento):
        print(f"P(pagar dívida da semana {s:>2})   {v:.2%}")
    print(f"P(pagar as três)              {r_final.prob_pagar_tudo:.2%}  (limite de risco {inst.limite_risco:.0%} -> "
          f"{'OK' if resultado['risco_dentro_do_limite'] else 'ACIMA DO LIMITE'})")
    print(f"Caixa final médio             R$ {final['caixa_final_medio']:,.2f}")
    print(f"Caixa final P{PERCENTIL_RISCO} (VaR 95%)      R$ {final[f'caixa_final_P{PERCENTIL_RISCO}']:,.2f}")
    print(f"Margem de contribuição        {final['margem_contrib_pct']:.1f}%")
    print(f"Estoque final médio           {final['estoque_final_unid']:.0f} unid de {int(inst.estoque.sum())}")
    print(f"Cenários de falha (base)      {falhas_base} de {args.cenarios}")
    print(f"Resultados salvos em {args.saida}")
    return resultado


if __name__ == "__main__":
    executar(parse_args())
