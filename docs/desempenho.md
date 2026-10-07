# Desempenho medido: tempo e memória

- **Como foi medido:** `comum/medicao.py::medir()` mede o tempo de parede com `time.perf_counter()` numa execução limpa. Depois repete a chamada com `tracemalloc` ativo para obter o **pico de memória**. As duas medições são separadas porque o tracemalloc distorce o tempo.
- **Ambiente:** Windows 10 Pro, Python 3.13.1, NumPy 2.4.4.
- **Como reproduzir:**

```bash
python questao_01/benchmark.py
python questao_02/benchmark.py
```

- **Arquivos brutos:** `questao_0X/resultados/desempenho_*.csv` e `otimizacao_*.csv`.

> Os tempos variam alguns milissegundos entre execuções, por causa da máquina. A ordem de grandeza e a forma de crescimento são estáveis.

## Questão 1: entradas pequena, média e grande

### Brute Force (crescimento exponencial)

| SKUs no subconjunto | Combinações | Tempo (s) | Memória de pico |
|---:|---:|---:|---:|
| 4 | 980 | 0,0005 | 0,3 MB |
| 5 | 3.920 | 0,0023 | 1,4 MB |
| 6 | 15.680 | 0,0144 | 6,6 MB |
| 7 | 78.400 | 0,0814 | 38 MB |
| **8 (usado no relatório)** | **548.800** | **0,70** | **111 MB** |

Cada SKU a mais multiplica o tempo por cerca de 6 a 9. O número de opções por SKU é `L − nivel_min_i`, que fica entre 4 e 7. Isso confirma O(Lⁿ). A memória para de crescer quando o lote enche (200 mil combinações). Para n ≥ 8, o pico é limitado por O(B_lote·n) e não por O(Lⁿ).

### GA (config A) e Greedy (crescimento polinomial)

| Entrada | SKUs | GA: avaliações (gerações) | GA: tempo | GA: memória | Greedy: tempo | Greedy: memória |
|---|---:|---:|---:|---:|---:|---:|
| pequena | 8 | 7.156 (72) | 0,023 s | 95 KB | 1,4 ms | 24 KB |
| média | 15 | 7.646 (77) | 0,027 s | 162 KB | 1,2 ms | 28 KB |
| grande (catálogo) | 22 | 8.920 (90) | 0,038 s | 229 KB | 1,2 ms | 29 KB |
| muito grande (catálogo ×2) | 44 | 9.998 (101) | 0,062 s | 437 KB | 1,3 ms | 38 KB |

A memória do GA cresce linearmente com n, como prevê O(P·n). O tempo cresce menos que linearmente porque cada geração é uma única operação NumPy sobre a matriz P×n, e o custo fixo por geração pesa mais que o custo por SKU nesses tamanhos. Em n = 22, o GA é **cerca de 18 vezes mais rápido** que o BF em n = 8 e resolve um espaço de 3,2·10¹³ combinações permitidas, cerca de 6·10⁷ vezes maior.

### Comparação das duas configurações do GA (catálogo completo, 5 seeds cada)

| Config | Fitness médio | Desvio | Melhor | Gerações | Tempo médio | Memória |
|---|---:|---:|---:|---:|---:|---:|
| A (P=100, torneio 3, uniforme, pm=1/n, elite 2) | 401.769,58 | **0,00** | 401.769,58 | 88 | 0,038 s | 229 KB |
| B (P=30, torneio 2, um ponto, pm=0,10, elite 1) | 401.347,23 | 400,61 | 401.769,58 | 215 | 0,058 s | 80 KB |

- **Convergência:** veja `questao_01/resultados/ga_fitness_por_geracao.png`. A config A chega ao mesmo ótimo nas 5 seeds e estabiliza antes da geração 40. A B usa 3,3 vezes menos memória, mas precisa de 2,4 vezes mais gerações e 1,5 vez mais tempo, e só acerta o ótimo em 1 das 5 seeds.
- **Por que B oscila:** a curva da média de B oscila muito, porque pm = 0,10 por gene (cerca de 2,2 genes mutados por filho) mantém a população dispersa, e o torneio binário quase não seleciona.
- **Por que A é melhor aqui:** a população maior e o crossover uniforme exploram melhor um problema com pouca dependência de posição. A ordem dos SKUs no cromossomo não tem significado, então o crossover de um ponto não tem "blocos" para preservar.

## Questão 2: escalabilidade (três tamanhos e estoque escalado)

| Entrada | SKUs | Estados da DP (Σ T·(S+1)) | MC construção (2000) | DP | Greedy | MC validação (5000) |
|---|---:|---:|---|---|---|---|
| pequena | 8 | 18.696 | 0,17 s · 19 MB | 0,020 s · 0,43 MB | 0,017 s · 0,08 MB | 0,070 s · 6,8 MB |
| média | 16 | 36.204 | 0,33 s · 39 MB | 0,040 s · 0,60 MB | 0,031 s · 0,13 MB | 0,130 s · 12,9 MB |
| grande | 24 | 55.116 | 0,50 s · 58 MB | 0,060 s · 0,85 MB | 0,044 s · 0,19 MB | 0,199 s · 18,9 MB |
| grande, estoque ×2 | 24 | 109.944 | 0,50 s · 58 MB | 0,091 s · 1,7 MB | 0,048 s · 0,18 MB | 0,194 s · 18,9 MB |
| grande, estoque ×4 | 24 | 219.600 | 0,50 s · 58 MB | 0,145 s · 3,3 MB | 0,044 s · 0,17 MB | 0,200 s · 18,9 MB |

- **Monte Carlo:** escala linearmente com n (8 → 24 SKUs: tempo e memória ×3) e não depende do estoque.
- **DP:** escala com o número de estados. A memória dobra quando o estoque dobra, como prevê O(n·T·S). O tempo cresce menos que linearmente porque a parte vetorizada sobre `s` dilui o custo fixo por estágio.
- **Greedy e validação:** não dependem de S.

### Número de cenários do Monte Carlo (24 SKUs)

| Cenários | Tempo | Memória de pico |
|---:|---:|---:|
| 1.000 | 0,26 s | 29 MB |
| 2.000 | 0,55 s | 58 MB |
| 5.000 | 1,34 s | 144 MB |
| 10.000 | 2,93 s | 289 MB |

Tempo e memória crescem de forma linear em N, cerca de 29 KB por cenário. O gargalo para escalar a estimativa é a memória. A alternativa está descrita em [complexidade.md](complexidade.md).

## Otimizações demonstradas (antes × depois, mesmo resultado numérico)

| Trecho | Antes | Depois | Tempo antes | Tempo depois | Speedup |
|---|---|---|---:|---:|---:|
| Brute Force, 6 SKUs (15.680 comb.) | `itertools.product` + laço Python (`brute_force_ingenuo`) | lotes NumPy com `unravel_index` (`brute_force`) | 0,196 s | 0,015 s | **13×** |
| Fitness de uma geração do GA (P=100, n=22) | laço Python por indivíduo (`fitness_ingenuo`) | matriz P×n + one-hot de categorias (`fitness`) | 8,6 ms | 0,35 ms | **24×** |
| DP do SKU Q2-017 (S=513) | Top-Down recursivo + `lru_cache` (`resolver_sku_top_down`) | Bottom-Up vetorizado em `s` (`resolver_sku`) | 0,219 s | 0,0045 s | **49×** |

- **Equivalência:** os testes `test_fitness_vetorizado_igual_ao_ingenuo`, `test_brute_force_vetorizado_igual_ao_ingenuo` e `test_dp_bottom_up_igual_top_down_e_reusa_subproblemas` garantem que as versões otimizadas produzem exatamente os mesmos valores.
- **Troca envolvida:** a versão vetorizada do BF **usa mais memória**: 6,6 MB, contra cerca de 1,4 KB do laço Python. Trocamos memória por tempo, e o tamanho do lote controla esse limite.
