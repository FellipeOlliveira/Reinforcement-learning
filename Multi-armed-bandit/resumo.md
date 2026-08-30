# Visão geral do projeto

Este projeto implementa e compara diferentes estratégias de escolha de ações no problema de **Multi-Armed Bandit**, utilizando um ambiente **Bernoulli Bandit** e um agente baseado na estratégia **ε-greedy**.

A ideia pode ser entendida como um conjunto de braços, onde cada braço possui uma probabilidade desconhecida de gerar uma recompensa. A cada passo, o agente precisa decidir qual braço escolher, tentando maximizar suas recompensas ao longo do tempo.

## Funcionamento geral

O sistema funciona através do seguinte ciclo:

1. **O agente escolhe um braço** com base nas estimativas que possui sobre cada braço e na estratégia de exploração utilizada.

2. **O ambiente recebe a ação** e retorna uma recompensa, que pode ser `0` ou `1`.

3. **O agente atualiza sua estimativa (`Q`)** sobre o braço escolhido utilizando a recompensa recebida.

4. **São registradas as métricas do experimento**, principalmente a recompensa obtida e o regret acumulado.

Esse processo é repetido durante vários passos para que seja possível observar como o agente aprende e como diferentes estratégias se comportam ao longo do tempo.

## Estratégias comparadas

O projeto compara três configurações do agente:

* **Greedy (ε = 0):** não realiza exploração. O agente sempre escolhe o braço que atualmente considera melhor.
* **ε-greedy (ε = 0.1):** realiza exploração em aproximadamente 10% das escolhas e, nos demais casos, escolhe o braço que considera melhor.
* **ε-greedy decrescente:** começa explorando bastante e reduz gradualmente o valor de ε ao longo do treinamento, passando de uma estratégia mais exploratória para uma estratégia mais gulosa.

Dessa forma, podemos comparar diferentes formas de equilibrar **exploração** (tentar novas possibilidades) e **aproveitamento** (utilizar o conhecimento já adquirido).

## Configuração do experimento

Para cada estratégia:

* São utilizados **10 braços**;
* O agente realiza **2000 passos**;
* O experimento é repetido **50 vezes**;
* Em cada repetição, as probabilidades verdadeiras dos braços são geradas novamente utilizando uma semente diferente;
* Os resultados das 50 execuções são posteriormente agregados por meio da média.

A repetição do experimento é importante porque existe aleatoriedade tanto na escolha das ações quanto nas recompensas recebidas. Ao utilizar várias execuções e calcular a média, conseguimos obter curvas mais estáveis e uma comparação mais confiável entre as estratégias.

## Métricas utilizadas

O projeto acompanha principalmente duas métricas:

### Recompensa média

Representa a recompensa obtida pelo agente ao longo dos passos. Quanto maior a recompensa média, melhor está sendo o desempenho do agente em aproveitar os braços com maiores probabilidades de recompensa.

### Regret acumulado

O regret representa a diferença entre o desempenho que poderia ser obtido escolhendo sempre o melhor braço e o desempenho obtido pelo agente.

Neste projeto, a cada passo é calculado:

**Regret = probabilidade do melhor braço − probabilidade do braço escolhido**

Esse valor é acumulado ao longo do experimento.

Assim, **quanto menor o regret acumulado, melhor**, pois significa que o agente está perdendo menos oportunidades de obter uma recompensa maior.

## Visualização dos resultados

Ao final do experimento são gerados dois gráficos:

1. **Recompensa média ao longo do tempo**, permitindo observar como cada estratégia evolui durante o aprendizado.
2. **Regret acumulado ao longo do tempo**, permitindo analisar quanto cada estratégia ficou distante do desempenho do melhor braço.

A partir desses gráficos, podemos avaliar qual estratégia apresenta melhor equilíbrio entre exploração e aproveitamento.

## Objetivo do projeto

O objetivo principal é **entender na prática o problema de exploração vs. aproveitamento no aprendizado por reforço**, implementando o ambiente, o agente e o processo de avaliação desde o início e comparando diferentes configurações de ε-greedy.
