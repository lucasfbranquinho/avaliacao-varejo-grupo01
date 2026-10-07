# Decisões de modelagem e justificativas

Este documento registra **por que** cada escolha foi feita. Também registra o que mudou ao longo do desenvolvimento quando os resultados mostraram problemas no modelo inicial (veja o histórico do Git).

## 0. Dados e integridade

- **Instância:** gerada pelo notebook oficial com **apenas o Bloco 02 alterado** (`GRUPO = "G01"`, `MATRICULAS = ["RM562262"]`). O resultado é `seed_base = 2` e `seed_final = 1000·1 + 2 = 1002`.
- **Arquivos oficiais:** os quatro CSV/JSON e o manifesto foram copiados sem edição para `data/`. Os nomes são os que o notebook produz (`q1_catalogo_G01.csv` etc.) e substituem o `instancia_grupoXX.csv` do exemplo de estrutura.
- **Verificação de integridade:** `comum/dados.py::verificar_integridade()` recalcula o SHA-256 de cada arquivo **toda vez que um `main.py` roda**. Se algo foi alterado, a execução é interrompida.
- **`.gitattributes`:** marca `data/**` como binário. Sem isso, o Git do Windows converteria CRLF em LF num clone e **mudaria os hashes**.
- **Cenários-surpresa:** são aplicados em memória por parâmetros de linha de comando. Nenhum arquivo de `data/` é tocado.

## 1. Questão 1: modelo econômico

### Caixa = margem de contribuição

O caixa por unidade é `u = p·(1 − taxa) − custo − logística`, a mesma fórmula que o notebook usa para calcular `meta_caixa_90d` e `divida_total_90d`. Usar outra definição (por exemplo, ignorar o custo do produto, já pago) tornaria a meta trivial. Mantivemos a convenção do gerador para que meta, dívida e resultado sejam comparáveis.

### Primeira versão e por que mudou

No modelo inicial, com demanda `D·m^(−ε)` simétrica, o ótimo era **todos os SKUs em +10%**. Isso acontecia mesmo com penalidade alta para o estoque parado (testamos perdas de 5% a 50% do custo). Com margens finas (u/p entre 5% e 35%), subir o preço sempre aumenta a contribuição, porque a perda de volume é pequena. O problema virava trivial e separável, e o GA não teria o que otimizar.

### Elasticidade assimétrica (efeito preço de referência)

`ε_aumento = 2 × ε` para m > 1. No varejo online, o cliente compara preços. Um aumento acima do preço de referência perde mais demanda do que um desconto equivalente ganha, um efeito documentado na literatura de *reference price* (loss aversion). Com isso, o ótimo passou a combinar níveis 1,00, 1,05 e 1,10 conforme a elasticidade e o estoque de cada SKU.

### Canibalização por categoria (γ = 0,5)

`d_i ∝ (m_i / m̄_categoria)^(−γ)`. Quando só um SKU da categoria desconta, ele "rouba" demanda dos vizinhos. Quando todos descontam igual, ninguém ganha participação. Esse termo **acopla os SKUs**: a melhor decisão para um depende da decisão dos outros. É isso que justifica BF e GA em vez de otimizar cada SKU isoladamente.

### Perda por obsolescência (25% do custo × idade/360)

Usa o campo `idade_estoque_dias`. O estoque que sobra depois de 90 dias perde valor, e mais ainda quando já é velho. Entra no fitness como penalidade e puxa a política para girar estoque antigo.

### Piso de margem

`u/p ≥ margem_minima_percentual` (8,72% na instância). Níveis que violam o piso são **proibidos por construção**: o gene só aceita níveis permitidos. A margem percentual cresce com o preço, então os níveis permitidos formam um sufixo `[nivel_min, L)`.

- **SKUs suspensos (8 de 30):** Q1-005, 009, 010, 014, 021, 023, 024 e 026 não respeitam o piso **nem a +10%**. Eles saem da política, porque vendê-los seria "liquidar abaixo do custo", exatamente o que o enunciado proíbe. O estoque deles entra no estoque remanescente do relatório.

### Teto anti-desconto indiscriminado

No máximo 40% dos SKUs (8 de 22) podem ficar com preço abaixo do atual. O GA garante isso com um **operador de reparo**. Na instância G01, o ótimo não usou nenhum desconto de preço. Isso é um resultado, não uma imposição: com estas margens, desconto de preço não gera caixa, e o estímulo de giro fica com os cupons do Greedy, pagos pela verba promocional.

### Escolha do subconjunto do BF

Os 8 SKUs de maior caixa na política atual. Eles cobrem 6 categorias, com 3 Eletrônicos e 2 Moda, então o acoplamento por canibalização aparece. Com 8 SKUs, são 548.800 combinações em 0,7 s. O enunciado pede pelo menos 6.

## 2. Questão 1: parâmetros do GA

| Parâmetro | Config A (principal) | Config B (comparação) | Justificativa |
|---|---|---|---|
| População | 100 | 30 | 100 cobre bem um espaço de 3,2·10¹³ com custo baixo (avaliação vetorizada); 30 testa economia de memória |
| Seleção | torneio k=3 | torneio k=2 | o torneio não depende da escala nem do sinal do fitness (a roleta depende); k=3 dá pressão moderada |
| Crossover | uniforme, pc=0,9 | um ponto, pc=0,9 | a ordem dos SKUs no cromossomo não tem significado, então não há "blocos" para o crossover de um ponto preservar |
| Mutação | reset do gene, pm=1/n (~1 gene por filho) | pm=0,10 (~2,2 genes por filho) | 1/n é a regra clássica; 0,10 testa mais exploração |
| Elitismo | 2 | 1 | garante que o melhor fitness nunca cai (histórico monotônico, verificado em teste) |
| Inicialização | aleatória + 1 indivíduo com a política atual | idem | a semente heurística garante que o GA nunca fica pior que "não fazer nada" |
| Parada | 300 gerações **ou** 60 sem melhora | idem | o gráfico mostra platô antes da geração 40 em A; 60 de paciência dá margem |

**Resultado:** A acerta o ótimo nas 5 seeds (desvio 0). B acerta em 1 das 5 e precisa de 2,4 vezes mais gerações. A tabela completa está em [desempenho.md](desempenho.md).

## 3. Questão 1: Greedy de cupons

- **O que é a verba:** a verba promocional (R$ 110.938,27) foi tratada como **verba de trade/cooperada**, dinheiro de fornecedores ou do marketplace que só pode ser usado em promoção. O cupom é pago pela verba, e o preço recebido pela empresa não muda.
- **Regra gulosa:** maior **benefício marginal por real investido**, por degraus de cupom (5, 10, 15 e 20%), usando uma fila de prioridade. Um SKU só sobe de degrau depois de comprar o anterior.
- **Resultado:** +R$ 41.701 de caixa com 99,5% da verba, contra +R$ 25.720 da linha de base "cupom uniforme de 5% para todos" (que gastou 93%). A concentração vence a distribuição uniforme.
- **Ressalva financeira:** cada R$ 1 de cupom devolve só **R$ 0,38** de caixa. Se a verba saísse do caixa da própria empresa, o cupom **destruiria** caixa (o caixa líquido cairia para R$ 373 mil). Recomendação: usar cupons apenas com verba de terceiros.

## 4. Questão 2: modelo estocástico

| Fonte de incerteza | Modelo | Justificativa |
|---|---|---|
| Demanda | lognormal de média 1, σ = `sigma_demanda`; 50% da variância é **comum** a todos os SKUs | um choque de mercado atinge o catálogo inteiro e cria risco de cauda para o caixa total; choques só independentes se cancelariam |
| Concorrente | passeio AR(1) em log, ρ = 0,7, σ = `sigma_preco_concorrente` | preços de concorrente têm persistência; ρ < 1 evita deriva infinita em 12 semanas |
| Logística | lognormal semanal comum, σ = `sigma_custo_logistico` | frete e combustível afetam todos os envios ao mesmo tempo |

- **Fórmula da demanda:** `λ = D·m^(−ε)·(r/r₀)^(−κ)`, com κ = 1,5 e r = preço/preço do concorrente. A mesma assimetria da Q1 vale para m > 1.
- **Arredondamento:** as vendas são inteiras, `min(estoque, round(λ·choque))`.

### Sobre a DP

- **Estado:** (semana, estoque do SKU), o mínimo pedido pelo enunciado. A DP é resolvida **SKU por SKU**, porque o estado conjunto teria cerca de 10⁵⁰ combinações. O caixa comum que paga as dívidas entra pelo **peso de urgência** `w_t = 1 + α·(fração da dívida que vence na semana t ou depois)`: caixa que chega cedo serve a mais vencimentos.
- **Calibração de α:** α ∈ {0; 0,25; 0,5; 1; 2} é escolhido pelo **MC de construção**. O critério é a maior P(pagar as 3) e, no empate, o maior P5 do caixa final. Resultado: α* = 0,5. Com a dívida base, todos os α pagam 100% e o desempate fica no P5. Com dívida ×1,6 (no MC de construção), P(pagar as 3) vai de 33,3% com α = 0 para 46,3% com α = 0,5, e cai para 30,6% com α = 2 (urgência demais vende barato cedo e falta margem na semana 12). A urgência importa justamente quando o caixa aperta, mas existe um ponto ótimo.
- **Faixas de quantis (B = 10):** as 2.000 amostras de cada (semana, SKU, nível) viram 10 faixas equiprováveis. É assim que **o Monte Carlo alimenta a DP**.
- **Ação PAUSAR e proibição de venda com prejuízo:** níveis com contribuição esperada ≤ 0 são proibidos. Na primeira versão, a DP vendia o Q2-011 abaixo do custo, porque perder 10% do custo em obsolescência parecia pior. Proibimos isso, por coerência com o piso de margem da Q1. Q2-011 e Q2-013 ficam pausados o horizonte inteiro.
- **Pausas pontuais:** em alguns casos (por exemplo, Q2-018 na semana 11) a DP pausa uma semana porque é indiferente vender na semana 11 ou na 12, já que todo o estoque sai de qualquer jeito. O empate é decidido por diferenças mínimas na logística média estimada.
- **Política em malha fechada:** na validação, cada cenário consulta `politica[i, t, s]` com o estoque **realizado**. O plano reage ao que acontece, em vez de seguir uma trajetória fixa.

### Sobre o Greedy da Q2 e o ajuste feito após a validação

- **Recurso:** até 6 destaques por semana (o recurso escasso), cada um com +25% de demanda no SKU.
- **Custo:** 3% da receita esperada do SKU na semana, com mínimo de R$ 150, dentro do orçamento semanal do config.
- **Benefício:** é calculado com a **função de valor V da DP**, não só pela venda extra imediata. Assim o Greedy não "compra" vendas que apenas seriam antecipadas, como as de um SKU que vai esgotar de qualquer forma.
- **O que a validação revelou:** na primeira versão (custo de 6% da receita e retorno mínimo 1,0), o Monte Carlo **independente** mostrou que os destaques **reduziam** o caixa final médio (R$ 209,0 mil para R$ 205,7 mil). Medimos o ganho real de cada destaque e encontramos dois problemas:
  1. a estimativa por faixas superestima o ganho em cerca de 10 a 20% em SKUs de giro baixo, por causa do arredondamento da demanda;
  2. com custo de 6% para +25% de demanda, o destaque ficava no ponto de equilíbrio (razão de 1,0 a 1,17).
- **Correção:** custo de 3%, mais próximo do ACoS típico de mídia em marketplace, e **retorno mínimo de 1,1** como margem de segurança contra o viés medido. Agora os destaques somam +R$ 8,0 mil no caixa médio e +R$ 10,2 mil no P5 da validação.
- **Por que isso importa:** é um exemplo de por que o enunciado exige validar o plano em cenários novos.

## 5. Interpretação financeira

### Questão 1 (Varejo Horizonte)

- **Caixa:** sobe de R$ 387,1 mil (preços atuais) para **R$ 483,7 mil**. Dessas, R$ 442,0 mil vêm da política do GA e +R$ 41,7 mil dos cupons do Greedy.
- **Meta:** R$ 316,9 mil, atingida com **folga de R$ 166,8 mil**. A dívida de 90 dias (R$ 298,4 mil) fica **162% coberta**.
- **Como o ganho vem:** de **subir preço onde a demanda aguenta** (13 SKUs a +10%, 2 a +5% e 7 mantidos no preço atual) e de **suspender 8 SKUs deficitários**. Não vem de desconto. A margem de contribuição fica em 22,7%.
- **Ponto de atenção:** sobram 773 unidades (23% do estoque inicial), que somam R$ 594 mil a custo. A maior parte está nos SKUs suspensos (eletrodomésticos e móveis de margem negativa). Isso exige outra solução fora da política de preço: devolução ao fornecedor, renegociação de custo ou outlet.
- **Cenários-surpresa** (em `questao_01/resultados/cenarios/`):

| Cenário | Caixa final | Folga sobre a meta | Cobertura da dívida |
|---|---:|---:|---:|
| base | R$ 483.682 | R$ 166.792 | 162% |
| meta +10% | R$ 483.682 | R$ 135.102 | 162% |
| logística +15% | R$ 469.740 | R$ 152.849 | 157% |
| verba −25% | R$ 474.549 | R$ 157.658 | 159% |
| sem o SKU Q1-016 (maior caixa) | R$ 405.631 | R$ 88.740 | 136% |
| GA com P=40, pm=0,05 | R$ 483.682 (mesmo ótimo) | R$ 166.792 | 162% |

A meta resiste a todos os cenários. O maior risco é **concentração**: perder o Q1-016 (Casa_Utilidades) sozinho tira R$ 78 mil.

### Questão 2 (Rede NovaChance)

- **Validação:** o plano DP + Greedy paga as três dívidas em **100% dos 5.000 cenários novos** (risco 0%, limite de 10%).
- **Caixa final:** média de R$ 217,0 mil e **P5 (VaR 95%) de R$ 199,1 mil**. Com preço fixo, seriam R$ 73,9 mil e R$ 59,4 mil.
- **Como a DP chega lá:**
  - **dá desconto** onde há excesso de estoque e demanda elástica (Q2-008 a 0,82 e Q2-022 a 0,88, ambos de Moda, com estoque zerando na semana 12);
  - **sobe o preço** em Beleza e Eletro, onde a demanda já esgota o estoque ou a margem é fina;
  - **pausa** os SKUs que dão prejuízo.
- **Margem:** a de contribuição sobe de 19,1% (preço fixo) para 26,1%.
- **Folga e ponto fraco:** o teste de estresse mostra quanto a dívida pode crescer mantendo o risco ≤ 10%. Com preço fixo, até ×1,17. Com a DP, ×1,53. Com DP + Greedy, ×1,555. Todas as falhas acontecem na **semana 12** (a maior parcela, R$ 163,6 mil). Os cenários de falha têm choque médio de demanda de 0,944, contra 1,003 nos de sucesso, e logística 2,7% mais cara. O risco é de **demanda fraca persistente**, não de concorrente.
- **Cenários-surpresa** (em `questao_02/resultados/cenarios/`):

| Cenário | P(3 dívidas), preço fixo | P(3 dívidas), DP + Greedy | Caixa final médio | P5 | Dívida suportável |
|---|---:|---:|---:|---:|---:|
| base | 100% | 100% | R$ 217.009 | R$ 199.051 | ×1,555 |
| concorrente −8% | 100% | 100% | R$ 177.218 | R$ 158.253 | ×1,445 |
| logística +15% | 100% | 100% | R$ 205.200 | R$ 186.689 | ×1,525 |
| verba −25% | 100% | 100% | R$ 217.009 | R$ 199.051 | ×1,555 |
| dívidas +10% | 100% | 100% | R$ 180.646 | R$ 162.687 | ×1,415 (sobre a nova dívida) |
| sem o SKU Q2-009 | **86,3%** | 100% | R$ 140.245 | R$ 123.187 | ×1,35 |
| concorrente −8% + logística +15% + dívidas +10% | **56,3%** | 100% | R$ 128.853 | R$ 109.720 | ×1,285 |

- **Verba −25% não muda nada:** o recurso que limita são as **6 vagas de destaque por semana** (todas ocupadas). A verba gasta, cerca de R$ 1,5 mil por semana, fica bem abaixo até do orçamento reduzido (R$ 4,3 mil por semana). Isso fica visível no relatório.
- **Onde o plano faz diferença:** nos cenários adversos, o preço fixo **viola o limite de risco**, enquanto o plano dinâmico mantém 100% de pagamento.
