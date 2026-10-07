"""Leitura da instância oficial do grupo e verificação de integridade (SHA-256).

Os arquivos de data/ NUNCA são alterados. Cenários-surpresa (meta +10%,
logística +15% etc.) são aplicados em memória, sobre cópias, por meio de
parâmetros de linha de comando.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PASTA_DADOS = RAIZ / "data"
GRUPO_PADRAO = "G01"

COLUNAS_Q1 = [
    "sku", "categoria", "custo_unitario", "preco_atual", "estoque",
    "demanda_base_90d", "elasticidade", "taxa_marketplace",
    "custo_logistico", "idade_estoque_dias",
]
COLUNAS_Q2 = [
    "sku", "categoria", "custo_unitario", "preco_atual", "estoque_inicial",
    "demanda_base_semanal", "elasticidade", "taxa_marketplace",
    "custo_logistico_base", "preco_concorrente_base",
]


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def verificar_integridade(pasta: Path = PASTA_DADOS) -> dict:
    """Recalcula os hashes e compara com o manifesto gerado pelo notebook.

    Retorna o manifesto. Lança ValueError se algum arquivo foi alterado.
    """
    manifesto = json.loads((pasta / "manifesto_dados.json").read_text(encoding="utf-8"))
    divergentes = [
        nome for nome, esperado in manifesto["arquivos"].items()
        if sha256_arquivo(pasta / nome) != esperado
    ]
    if divergentes:
        raise ValueError(f"Arquivos alterados após a geração: {divergentes}")
    return manifesto


def _validar_colunas(df: pd.DataFrame, colunas: list[str], nome: str) -> None:
    faltando = [c for c in colunas if c not in df.columns]
    if faltando:
        raise ValueError(f"{nome}: colunas ausentes {faltando}")
    if df["sku"].duplicated().any():
        raise ValueError(f"{nome}: SKU duplicado")
    numericas = [c for c in colunas if c not in ("sku", "categoria")]
    if (df[numericas] < 0).any().any():
        raise ValueError(f"{nome}: valores negativos em colunas numéricas")


def carregar_q1(pasta: Path = PASTA_DADOS, grupo: str = GRUPO_PADRAO, verificar: bool = True):
    """Retorna (catalogo DataFrame, config dict) da Questão 1."""
    if verificar:
        verificar_integridade(pasta)
    cat = pd.read_csv(pasta / f"q1_catalogo_{grupo}.csv", encoding="utf-8-sig")
    cfg = json.loads((pasta / f"q1_config_{grupo}.json").read_text(encoding="utf-8"))
    _validar_colunas(cat, COLUNAS_Q1, "Q1")
    return cat, cfg


def carregar_q2(pasta: Path = PASTA_DADOS, grupo: str = GRUPO_PADRAO, verificar: bool = True):
    """Retorna (catalogo DataFrame, config dict) da Questão 2."""
    if verificar:
        verificar_integridade(pasta)
    cat = pd.read_csv(pasta / f"q2_catalogo_{grupo}.csv", encoding="utf-8-sig")
    cfg = json.loads((pasta / f"q2_config_{grupo}.json").read_text(encoding="utf-8"))
    _validar_colunas(cat, COLUNAS_Q2, "Q2")
    return cat, cfg


def remover_skus(cat: pd.DataFrame, skus: list[str] | None) -> pd.DataFrame:
    """Cenário-surpresa: SKUs que deixam de estar disponíveis."""
    if not skus:
        return cat
    desconhecidos = set(skus) - set(cat["sku"])
    if desconhecidos:
        raise ValueError(f"SKU inexistente: {sorted(desconhecidos)}")
    return cat[~cat["sku"].isin(skus)].reset_index(drop=True)
