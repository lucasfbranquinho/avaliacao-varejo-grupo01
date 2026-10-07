# Complexidade assintótica (Big O) — tempo e memória

Todas as expressões abaixo usam variáveis com significado explícito e apontam o trecho de código que domina o custo. As medições reais que confirmam cada expressão estão em [desempenho.md](desempenho.md).

## Variáveis

| Símbolo | Significado | Valor na instância G01 |
|---|---|---|
| `n` | SKUs ativos (Q1: após suspender os que não respeitam a margem mínima) | Q1: 22 (de 30) · Q2: 24 |
| `L` | níveis de preço da Q1 | 7 |
| `c` | categorias distintas (one-hot da canibalização) | ≤ 7 |
| `P` | tamanho da população do GA | A: 100 · B: 30 |
| `G` | gerações executadas pelo GA (até `geracoes_max` ou `paciencia`) | ~70–300 |
| `J` | degraus de cupom do Greedy da Q1 | 4 |
| `T` | semanas da Q2 | 12 |
| `K` | ações da DP: 6 níveis de preço + PAUSAR | 7 |
| `S` | estoque inicial de um SKU = estados de estoque da DP (`0..S`) | até 513 |
| `B` | faixas de quantis da demanda (saída do Monte Carlo) | 10 |
| `N` | cenários Monte Carlo | 2.000 (construção) · 5.000 (validação) |

---

## Questão 1

### A. Brute Force — [`questao_01/brute_force.py`](../questao_01/brute_force.py) · `brute_force()`

- **Tempo: O(Lⁿ · n · c)**, ou seja, O(Lⁿ · n) com `c` limitado por 7. O laço `for ini in range(0, total, lote)` percorre as `N_comb = ∏ (L − nivel_min_i) ≤ Lⁿ` combinações. Cada uma é avaliada por `modelo.avaliar()` em O(n·c), por causa do produto `m @ onehot_cat`.
- **Memória: O(B_lote · n)**. Só um lote de `B_lote = 200.000` combinações fica na memória por vez (`np.unravel_index` gera o lote sob demanda). Sem lotes, seriam O(Lⁿ · n).
- **Trecho dominante:** `fitness(inst, x)` dentro do laço de lotes.
- **Quando deixa de ser viável:** cada SKU a mais multiplica o custo por cerca de 7. Medimos 0,70 s para n = 8 (548.800 combinações). Para n = 10 seriam cerca de 30 s, para n = 12 cerca de 28 min, e o catálogo inteiro (n = 22) tem 3,2·10¹³ combinações permitidas após o piso de margem (o teto teórico é 7²² ≈ 3,9·10¹⁸). Isso levaria cerca de 1,3 ano no ritmo medido de ~780 mil combinações por segundo. Por isso o código recusa instâncias acima de `limite = 50 milhões` de combinações.
- **Alternativa adotada:** o GA, validado contra o BF em n = 8 com gap de 0%.

### B. Algoritmo Genético — [`questao_01/genetic_algorithm.py`](../questao_01/genetic_algorithm.py) · `algoritmo_genetico()`

- **Tempo: O(G · P · n · c)**. Cada geração faz:
  - torneio O(P·k);
  - crossover e mutação O(P·n);
  - reparo O(P·n);
  - avaliação vetorizada `_avaliar()` O(P·n·c).

  A avaliação domina o custo.
- **Memória: O(P · n + G)**. São a população, os filhos e as máscaras (matrizes P×n), mais os dois históricos de tamanho G usados no gráfico de convergência.
- **Trecho dominante:** `fit = np.concatenate([..., _avaliar(inst, filhos)])`.
- **Viabilidade:** é linear em n. Medimos n = 44 (catálogo duplicado) em 0,06 s. O limite prático é o número de avaliações, não a memória. Para catálogos de milhares de SKUs, basta reduzir `P` ou `G` (o gráfico mostra que a config A converge em menos de 40 gerações). Outra opção é decompor o problema por categoria, já que só há acoplamento dentro da categoria.

### C. Greedy de cupons — [`questao_01/greedy.py`](../questao_01/greedy.py) · `alocar_incentivos()`

- **Tempo: O(n·J + n·J·log n)**. `_tabela_degraus()` monta a tabela de vendas e gasto por degrau em O(n·J). O heap começa com n candidatos, e cada um dos no máximo n·J `heappop` e `heappush` custa O(log n).
- **Memória: O(n · J)**, para as tabelas `q` e `gasto` e o heap.
- **Trecho dominante:** o laço `while heap:`.
- **Viabilidade:** sempre viável. Medimos 1,3 ms para n = 44. É a etapa mais barata do pipeline.

**Custo total da Q1:** O(Lⁿˢ·nₛ + R·G·P·n + n·J·log n), em que nₛ é o tamanho do subconjunto do BF e R é o número de repetições do GA. O termo exponencial do BF é controlado mantendo nₛ ≤ 8.

---

## Questão 2

### A/D. Monte Carlo — [`questao_02/monte_carlo.py`](../questao_02/monte_carlo.py)

| Função | Tempo | Memória | Trecho dominante |
|---|---|---|---|
| `gerar_cenarios()` | O(N·T·n) | O(N·T·n) | sorteios normais e laço AR(1) do concorrente (`for t in range(1, T)`) |
| `estimar_estatisticas()` | **O(K · N·T·n · log N)** | O(N·T·n) por nível | `d.sort(axis=0)`, que ordena as N amostras de cada (t, i) para extrair os quantis |
| `simular()` (validação) | O(N·T·n) | O(N·T·n) | gather `politica[sku, t, s]` e cálculo das vendas por semana |

- Na validação, a memória O(N·T·n) vem de guardar `estoque_semana` e `nivel_semana` para os gráficos de trajetória. Sem esses registros, seria O(N·n).
- **Viabilidade:** é linear em N (medido de 1.000 a 10.000 cenários: 0,26 → 2,9 s e 29 → 289 MB). O fator que limita é a **memória**: 2.000 cenários ocupam cerca de 58 MB no pico. Para 100 mil cenários, a alternativa é processar em lotes de cenários e acumular os quantis por histograma, como já fazemos no BF da Q1.

### B. Programação Dinâmica — [`questao_02/dynamic_programming.py`](../questao_02/dynamic_programming.py) · `resolver_sku()`

- **Tempo: O(n · T · S · K · B)**. Para cada SKU (n) e semana (T), o código avalia K ações sobre todos os S+1 estados, e cada avaliação faz a média sobre B faixas: `np.minimum(s[:, None], d[t, k][None, :])` e `V[t + 1][s[:, None] - vend]`.
- **Memória: O(n · T · S)** para a tabela de política `politica[n, T, Smax+1]` (necessária na execução em malha fechada), mais O(T·S) por SKU para `V` e O(K·S·B) temporários.
- **Reutilização de subproblemas:** `V[t+1]` é calculado uma vez e consultado por todos os pares (s, k, b) da semana t. No SKU Q2-017 (S = 513), a versão Top-Down com `lru_cache` registrou **281.751 reusos (hits) para apenas 5.194 subproblemas distintos**, ou seja, cerca de 54 consultas por subproblema.
- **Sem a DP:** enumerar decisões em árvore custaria O((K·B)ᵀ) = 70¹² ≈ 1,4·10²² por estado inicial.
- **Por que a DP é feita por SKU:** a DP conjunta teria ∏(Sᵢ+1) ≈ 10⁵⁰ estados. O acoplamento entre SKUs (o caixa comum) entra pelos pesos de urgência `w_t` e é conferido na validação Monte Carlo.
- **Viabilidade:** é linear em S. Com estoque ×4 (219.600 estados), o tempo foi de 0,14 s e o pico de memória de 3,3 MB. A DP deixaria de caber se o estado incluísse o caixa (estado (t, s, caixa) com caixa discretizado em R$ 1 mil, cerca de 500 valores, multiplicaria tempo e memória por 500) ou se fosse conjunta entre SKUs. Nesses casos, a alternativa seria agregação de estados ou aproximação do valor.

### C. Greedy de destaques — [`questao_02/greedy.py`](../questao_02/greedy.py) · `alocar_destaques()`

- **Tempo: O(T · n · (B + log n))**. Para cada semana, cada SKU tem o benefício calculado em O(B) com o `V` da DP, e a ordenação dos candidatos custa O(n log n).
- **Memória: O(T · n)**, para as matrizes de destaques e custos e a tabela de escolhas.
- **Trecho dominante:** o cálculo de `ganho`, que usa a tabela `V1[s - v2]`.
- **Viabilidade:** sempre viável (44 ms medidos, quase todo o tempo em `pandas .loc`).

**Custo total da Q2:** O(K·N_c·T·n·log N_c + |α|·(n·T·S·K·B + N_c·T·n) + N_v·T·n). O termo |α| vem da calibração: cada um dos 5 valores de α roda DP, Greedy e uma simulação no MC de construção. Na prática, o Monte Carlo domina (cerca de 0,5 s de construção e 0,2 s de validação, contra 0,06 s de uma DP).
