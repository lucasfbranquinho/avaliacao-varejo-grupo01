"""Testes da Questão 1: casos pequenos, limites, entradas inválidas e funções principais."""

import itertools

import numpy as np
import pandas as pd
import pytest

from comum.dados import carregar_q1, remover_skus, verificar_integridade
from questao_01.brute_force import brute_force, brute_force_ingenuo, n_combinacoes
from questao_01.genetic_algorithm import CONFIG_A, ConfigGA, algoritmo_genetico
from questao_01.greedy import alocar_incentivos
from questao_01.modelo import (ParametrosQ1, avaliar, construir_instancia, fitness,
                               fitness_ingenuo, subinstancia)

NIVEIS = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]


def catalogo_pequeno():
    return pd.DataFrame({
        "sku": ["A", "B", "C", "D"],
        "categoria": ["X", "X", "Y", "Y"],
        "custo_unitario": [50.0, 80.0, 30.0, 400.0],
        "preco_atual": [100.0, 120.0, 90.0, 410.0],   # D é deficitário em qualquer nível
        "estoque": [100, 60, 200, 10],
        "demanda_base_90d": [80, 70, 150, 8],
        "elasticidade": [1.5, 1.2, 2.0, 1.0],
        "taxa_marketplace": [0.10, 0.12, 0.15, 0.10],
        "custo_logistico": [5.0, 8.0, 4.0, 20.0],
        "idade_estoque_dias": [30, 200, 400, 10],
    })


CFG = {"niveis_preco_relativos": NIVEIS, "margem_minima_percentual": 0.08}


@pytest.fixture
def inst():
    return construir_instancia(catalogo_pequeno(), CFG)


# ---------------------------------------------------------------- dados ----
def test_integridade_dos_dados_oficiais():
    manifesto = verificar_integridade()
    assert manifesto["grupo"] == "G01" and manifesto["seed_final"] == 1002


def test_carregar_q1_oficial():
    cat, cfg = carregar_q1()
    assert len(cat) == 30 and cfg["niveis_preco_relativos"] == NIVEIS


def test_remover_sku_inexistente_falha():
    with pytest.raises(ValueError):
        remover_skus(catalogo_pequeno(), ["NAO_EXISTE"])


# --------------------------------------------------------------- modelo ----
def test_sku_deficitario_e_suspenso(inst):
    assert inst.suspensos == ["D"] and inst.n == 3


def test_piso_de_margem_respeitado(inst):
    for i in range(inst.n):
        assert inst.margem_pct[i, inst.nivel_min[i]] >= CFG["margem_minima_percentual"]
        if inst.nivel_min[i] > 0:
            assert inst.margem_pct[i, inst.nivel_min[i] - 1] < CFG["margem_minima_percentual"]


def test_vendas_nunca_excedem_estoque(inst):
    todos = np.array(list(itertools.product(*[range(a, 7) for a in inst.nivel_min])))
    r = avaliar(inst, todos)
    assert (r["vendas"] <= inst.estoque).all() and (r["vendas"] >= 0).all()


def test_fitness_vetorizado_igual_ao_ingenuo(inst):
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = inst.nivel_min + (rng.random(inst.n) * inst.n_opcoes).astype(int)
        assert fitness(inst, x)[0] == pytest.approx(fitness_ingenuo(inst, x))


def test_margem_minima_invalida():
    with pytest.raises(ValueError):
        construir_instancia(catalogo_pequeno(), {**CFG, "margem_minima_percentual": 1.5})


def test_teto_de_descontos_marca_inviavel():
    inst = construir_instancia(catalogo_pequeno(), CFG, ParametrosQ1(max_frac_desconto=0.0))
    x = inst.nivel_padrao().copy()
    assert avaliar(inst, x)["viavel"][0]
    i = int(np.argmin(inst.nivel_min))
    assert inst.nivel_min[i] < inst.idx_nivel_atual   # existe SKU que pode descontar
    x[i] = inst.nivel_min[i]                          # um único desconto já viola o teto 0
    assert not avaliar(inst, x)["viavel"][0]
    assert fitness(inst, x)[0] == -np.inf


# ---------------------------------------------------------- brute force ----
def test_brute_force_conta_todas_as_combinacoes(inst):
    assert brute_force(inst).combinacoes == n_combinacoes(inst) == int(np.prod(inst.n_opcoes))


def test_brute_force_vetorizado_igual_ao_ingenuo(inst):
    a, b = brute_force(inst, lote=7), brute_force_ingenuo(inst)
    assert a.fitness == pytest.approx(b.fitness) and a.viaveis == b.viaveis


def test_brute_force_recusa_instancia_grande(inst):
    with pytest.raises(ValueError):
        brute_force(inst, limite=10)


# ------------------------------------------------------------------- GA ----
def test_ga_reproduzivel_por_seed(inst):
    a, b = algoritmo_genetico(inst, CONFIG_A, seed=42), algoritmo_genetico(inst, CONFIG_A, seed=42)
    assert a.fitness == b.fitness and (a.niveis == b.niveis).all()


def test_ga_atinge_otimo_do_bf_em_instancia_pequena(inst):
    assert algoritmo_genetico(inst, CONFIG_A, seed=1).fitness == pytest.approx(brute_force(inst).fitness)


def test_ga_historico_monotono_e_politica_viavel(inst):
    g = algoritmo_genetico(inst, CONFIG_A, seed=3)
    assert all(b >= a for a, b in zip(g.historico_melhor, g.historico_melhor[1:]))
    assert avaliar(inst, g.niveis)["viavel"][0]


def test_ga_config_invalida():
    with pytest.raises(ValueError):
        ConfigGA(populacao=2, elitismo=2).validar()
    with pytest.raises(ValueError):
        ConfigGA(taxa_mutacao=1.5).validar()


def test_ga_no_catalogo_oficial_valida_contra_bf():
    cat, cfg = carregar_q1()
    inst = construir_instancia(cat, cfg)
    sub = subinstancia(inst, range(6))
    bf = brute_force(sub)
    ga = algoritmo_genetico(sub, CONFIG_A, seed=cfg["seed_final"])
    assert ga.fitness <= bf.fitness + 1e-6
    assert (bf.fitness - ga.fitness) / bf.fitness < 0.01


# --------------------------------------------------------------- greedy ----
def test_greedy_respeita_orcamento(inst):
    x = inst.nivel_padrao()
    for orc in [0.0, 50.0, 500.0, 1e9]:
        r = alocar_incentivos(inst, x, orc)
        assert r.gasto.sum() <= orc + 1e-6
        assert (r.vendas <= inst.estoque).all()
        assert r.fitness_depois >= r.fitness_antes - 1e-6


def test_greedy_orcamento_zero_nao_muda_nada(inst):
    r = alocar_incentivos(inst, inst.nivel_padrao(), 0.0)
    assert r.caixa_depois == pytest.approx(r.caixa_antes) and (r.cupom == 0).all()


def test_greedy_orcamento_negativo_falha(inst):
    with pytest.raises(ValueError):
        alocar_incentivos(inst, inst.nivel_padrao(), -1.0)


def test_cenario_logistica_maior_reduz_caixa():
    cat, cfg = carregar_q1()
    base = construir_instancia(cat, cfg)
    caro = construir_instancia(cat, cfg, ParametrosQ1(logistica_mult=1.15))
    assert caro.contrib.sum() < base.contrib.sum()
