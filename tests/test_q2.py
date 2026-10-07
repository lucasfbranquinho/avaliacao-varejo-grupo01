"""Testes da Questão 2: Monte Carlo, DP (Bottom-Up x Top-Down), Greedy e validação."""

import numpy as np
import pandas as pd
import pytest

from comum.dados import carregar_q2
from questao_02.dynamic_programming import (programacao_dinamica, resolver_sku, resolver_sku_top_down,
                                            trajetoria_esperada)
from questao_02.greedy import alocar_destaques
from questao_02.modelo import ParametrosQ2, construir_instancia
from questao_02.monte_carlo import (SEED_OFFSET_VALIDACAO, estimar_estatisticas, gerar_cenarios,
                                    politica_fixa, simular)

CFG = {
    "numero_semanas": 6, "caixa_inicial": 1000.0,
    "dividas": {"semana_2": 500.0, "semana_4": 800.0, "semana_6": 1200.0},
    "variabilidade": {"sigma_demanda": 0.15, "sigma_preco_concorrente": 0.05, "sigma_custo_logistico": 0.10},
    "orcamento_promocional_semanal": 300.0, "niveis_preco_relativos": [0.82, 0.88, 0.94, 1.00, 1.06, 1.12],
    "n_simulacoes_recomendadas": 200, "limite_risco_falha": 0.10,
}


def catalogo_pequeno():
    return pd.DataFrame({
        "sku": ["A", "B", "C"], "categoria": ["Moda", "Beleza", "Eletro"],
        "custo_unitario": [40.0, 20.0, 950.0], "preco_atual": [100.0, 60.0, 1000.0],  # C: prejuízo sempre
        "estoque_inicial": [30, 25, 5], "demanda_base_semanal": [6, 5, 1],
        "elasticidade": [1.8, 1.2, 1.0], "taxa_marketplace": [0.10, 0.12, 0.10],
        "custo_logistico_base": [8.0, 5.0, 90.0], "preco_concorrente_base": [95.0, 62.0, 1010.0],
    })


@pytest.fixture
def base():
    inst = construir_instancia(catalogo_pequeno(), CFG)
    cen = gerar_cenarios(inst, 200, seed=7)
    return inst, cen, estimar_estatisticas(inst, cen)


# ---------------------------------------------------------------- Monte Carlo ----
def test_mc_reproduzivel_e_seeds_diferentes(base):
    inst, cen, _ = base
    igual = gerar_cenarios(inst, 200, seed=7)
    outro = gerar_cenarios(inst, 200, seed=7 + SEED_OFFSET_VALIDACAO)
    assert np.array_equal(cen.choque_dem, igual.choque_dem) and np.array_equal(cen.pc, igual.pc)
    assert not np.allclose(cen.choque_dem, outro.choque_dem)


def test_choque_de_demanda_tem_media_proxima_de_1():
    inst = construir_instancia(catalogo_pequeno(), CFG)
    cen = gerar_cenarios(inst, 20000, seed=1)
    assert cen.choque_dem.mean() == pytest.approx(1.0, abs=0.01)
    assert (cen.logistica / inst.logistica).mean() == pytest.approx(1.0, abs=0.01)


def test_mc_zero_cenarios_falha(base):
    with pytest.raises(ValueError):
        gerar_cenarios(base[0], 0, seed=1)


def test_bins_ordenados_e_demanda_cai_com_preco(base):
    inst, _, est = base
    assert (np.diff(est.demanda_bins, axis=-1) >= -1e-9).all()               # quantis crescentes
    media = est.demanda_bins.mean(axis=-1)                                    # (T, n, K)
    assert (np.diff(media, axis=-1) <= 1e-9).all()                            # preço maior => menos demanda


# ------------------------------------------------------------------------- DP ----
def test_dp_bottom_up_igual_top_down_e_reusa_subproblemas(base):
    inst, _, est = base
    pesos = inst.pesos_urgencia(0.5)
    for i in range(inst.n):
        V, _ = resolver_sku(inst, est, i, pesos)
        valor, cache = resolver_sku_top_down(inst, est, i, pesos)
        assert V[0, inst.estoque[i]] == pytest.approx(valor)
        if inst.skus[i] != "C":                   # C só pode pausar: um único caminho, sem reuso
            assert cache.hits > cache.misses        # cada subproblema é consultado várias vezes


def test_dp_nao_vende_com_prejuizo(base):
    inst, _, est = base
    dp = programacao_dinamica(inst, est, 0.0)
    assert (dp.politica[2, :, 1:inst.estoque[2] + 1] == inst.PAUSAR).all()   # SKU C sempre pausado


def test_dp_estoque_zero_vale_zero(base):
    inst, _, est = base
    for V in programacao_dinamica(inst, est, 1.0).valor:
        assert V[:, 0] == pytest.approx(0.0)


def test_dp_valor_monotono_no_estoque(base):
    """Ter mais estoque nunca piora o valor ótimo na semana 0 (pode-se pausar)... exceto pela perda
    terminal; com perda zero, V_0 deve ser não-decrescente em s."""
    inst = construir_instancia(catalogo_pequeno(), CFG, ParametrosQ2(perda_obsolescencia=0.0))
    est = estimar_estatisticas(inst, gerar_cenarios(inst, 200, seed=3))
    for V in programacao_dinamica(inst, est).valor:
        assert (np.diff(V[0]) >= -1e-9).all()


def test_trajetoria_reconstruida_consistente(base):
    inst, _, est = base
    dp = programacao_dinamica(inst, est, 0.5)
    plano = trajetoria_esperada(inst, est, dp)
    assert len(plano) == inst.n * inst.T
    for sku, g in plano.groupby("sku"):
        assert (g["estoque_fim"] >= 0).all()
        assert (g["estoque_inicio"].iloc[1:].to_numpy() == g["estoque_fim"].iloc[:-1].to_numpy()).all()


# --------------------------------------------------------------------- Greedy ----
def test_greedy_respeita_orcamento_e_vagas(base):
    inst, _, est = base
    dp = programacao_dinamica(inst, est, 0.5)
    plano = trajetoria_esperada(inst, est, dp)
    for orc in [0.0, 100.0, 1e6]:
        g = alocar_destaques(inst, est, dp, plano, orcamento_semanal=orc)
        assert (g.gasto_semanal <= orc + 1e-6).all()
        assert (g.destaques.sum(axis=1) <= inst.params.max_destaques_semana).all()
        assert not g.destaques[:, 2].any()                                    # SKU pausado não é destacado
    assert alocar_destaques(inst, est, dp, plano, orcamento_semanal=0.0).destaques.sum() == 0


def test_greedy_orcamento_negativo_falha(base):
    inst, _, est = base
    dp = programacao_dinamica(inst, est)
    with pytest.raises(ValueError):
        alocar_destaques(inst, est, dp, trajetoria_esperada(inst, est, dp), orcamento_semanal=-1)


# ------------------------------------------------------------------ Simulação ----
def test_simulacao_conserva_caixa_e_estoque(base):
    inst, cen, est = base
    dp = programacao_dinamica(inst, est, 0.5)
    r = simular(inst, cen, dp.politica)
    assert (r.estoque_final >= 0).all() and (r.estoque_final <= inst.estoque).all()
    esperado = inst.caixa_inicial + r.contribuicao - inst.dividas.sum()
    assert np.allclose(r.caixa_final, esperado)
    assert 0.0 <= r.prob_pagar_tudo <= 1.0


def test_dp_supera_preco_fixo_no_caixa(base):
    inst, cen, est = base
    dp = programacao_dinamica(inst, est, 0.0)
    cv = gerar_cenarios(inst, 500, seed=99)
    assert simular(inst, cv, dp.politica).caixa_final.mean() >= simular(inst, cv, politica_fixa(inst, 3)).caixa_final.mean()


def test_divida_fora_do_horizonte_falha():
    cfg = {**CFG, "dividas": {"semana_9": 100.0}}
    with pytest.raises(ValueError):
        construir_instancia(catalogo_pequeno(), cfg)


def test_instancia_oficial_carrega():
    cat, cfg = carregar_q2()
    inst = construir_instancia(cat, cfg)
    assert inst.n == 24 and inst.T == 12 and list(inst.semanas_divida) == [4, 8, 12]
