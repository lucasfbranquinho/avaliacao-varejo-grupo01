# Registro de uso de IA generativa

Conforme a seção 3 do enunciado, este arquivo registra como a IA generativa foi usada neste checkpoint.

## Ferramenta

- **Claude Code** (Anthropic), modelo Claude Opus 5.5, usado em outubro de 2026.

## O que a IA fez

O uso foi **extenso**: não se limitou a revisão de sintaxe. A ferramenta trabalhou sob a orientação do aluno e:

- executou o notebook gerador, depois de alterar apenas o Bloco 02 com o grupo e a matrícula informados pelo aluno;
- propôs a modelagem econômica das duas questões e escreveu o código de `comum/`, `questao_01/` e `questao_02/`;
- escreveu os testes de `tests/`;
- rodou os experimentos e benchmarks;
- redigiu a documentação de `docs/` e este README.

## Decisões em que a validação dos resultados mudou o modelo

Estas decisões estão registradas no histórico do Git e em `docs/decisoes.md`:

- Assimetria de elasticidade na Q1, adotada depois de verificar que o modelo simétrico levava a "todos os preços em +10%".
- Proibição de venda com contribuição negativa na DP da Q2.
- Recalibração do custo do destaque e margem de segurança no Greedy da Q2, depois que a validação Monte Carlo mostrou que os destaques estavam reduzindo o caixa.

## Responsabilidade do aluno

O enunciado exige que o código, as decisões e os resultados possam ser explicados e modificados pelo próprio aluno. Para isso, antes da defesa o aluno deve:

- estudar o fluxo de cada questão (`main.py`), as formulações em `docs/decisoes.md` e as complexidades em `docs/complexidade.md`;
- executar os cenários-surpresa da tabela do README e entender por que cada indicador muda.

### Pontos de revisão do aluno

_Preencher antes da entrega: o que foi revisado, entendido, alterado ou questionado._

-
