# Modelo híbrido com fusão adaptativa otimizada por PSO

O modelo combina as probabilidades de **SVM RBF, Random Forest e k-NN**. Durante o treinamento, o PSO aprende seis coeficientes a partir de predições out-of-fold dos especialistas. Na inferência, esses coeficientes e a confiança dos especialistas determinam os três pesos de cada nova amostra.

[Diagrama editável no Draw.io](modelo%20hibrido.drawio) · [Visualização do diagrama](modelo%20hibrido.drawio.png) · [Perguntas para a apresentação](README_PERGUNTAS_PROFESSOR.md)

## Componentes

| Componente | O que faz |
| --- | --- |
| Pré-processamento | Cada especialista recebe seu próprio pipeline. Para atributos numéricos, usa imputação pela mediana e padronização. Para categóricos, usa imputação pela moda e one-hot encoding. |
| SVM RBF | Produz probabilidades de classe com `SVC(kernel="rbf", probability=True)`. |
| Random Forest | Produz probabilidades com 100 árvores. |
| k-NN | Produz probabilidades a partir de cinco vizinhos. |
| PSO | Aprende um coeficiente de importância e um coeficiente de influência da confiança para cada especialista. |
| Fusão adaptativa | Converte confiança e coeficientes em pesos com softmax, combina as probabilidades e escolhe a classe de maior probabilidade. |

O benchmark constrói o pré-processador. A classe `PSOAdaptiveFusionClassifier` clona esse transformador para cada especialista. [Pré-processamento](../src/02_run_benchmark.py#L50) · [Implementação dos especialistas](../src/pso_adaptive_fusion.py#L157)

## Treinamento

1. **Separar treino e teste externos.** O benchmark usa cinco folds estratificados, com embaralhamento e semente 42. Cada fit do híbrido recebe apenas o treino do fold externo. [Benchmark](../src/02_run_benchmark.py#L139)
2. **Produzir probabilidades OOF.** Dentro do treino externo, o modelo monta três folds estratificados sem embaralhamento. Cada especialista ajusta um clone do pipeline no treino interno e prevê as probabilidades na respectiva validação. Cada amostra recebe, de cada especialista, a previsão de um pipeline ajustado sem aquela amostra. O tensor resultante tem dimensão $N \times 3 \times C$, em que $N$ é o número de amostras de treino externo e $C$ é o número de classes. [Geração OOF](../src/pso_adaptive_fusion.py#L46) · [Ajuste do híbrido](../src/pso_adaptive_fusion.py#L138)
3. **Otimizar a fusão.** O PSO avalia cada conjunto candidato de coeficientes usando o mesmo tensor OOF. Calcula os pesos, funde as probabilidades, escolhe as classes por `argmax` e compara as decisões com os rótulos do treino. Maximiza o F1 Macro sobre todas as predições OOF reunidas. Em caso de empate, prefere a menor média dos quadrados dos coeficientes. [Função objetivo e ciclo do PSO](../src/pso_adaptive_fusion.py#L56)
4. **Ajustar os pipelines finais.** Após a busca, os três pipelines são ajustados com todo o treino externo. O modelo guarda os pipelines em `estimators_`, os coeficientes em `coefficients_` e a ordem dos rótulos em `classes_`. [Ajuste final](../src/pso_adaptive_fusion.py#L181)

A proteção contra vazamento vem do escopo do ajuste de cada pipeline. A imputação, a escala e as categorias de um fold interno são aprendidas apenas no treino interno correspondente. O teste externo entra na predição de avaliação depois do ajuste. [Construção dos pipelines](../src/pso_adaptive_fusion.py#L157) · [Treino e avaliação no benchmark](../src/02_run_benchmark.py#L189)

## Inferência

1. A amostra $x$ passa pelos três pipelines ajustados. Cada pipeline retorna seu vetor de probabilidades $p_i(x)$.
2. O modelo calcula a confiança $h_i(x)$ de cada especialista e aplica os coeficientes aprendidos para obter os pesos $w_i(x)$.
3. A média ponderada produz a distribuição final. `predict_proba()` retorna essa distribuição. `predict()` aplica `argmax` e recupera o rótulo em `classes_`.

A busca PSO fica no treinamento. Os **coeficientes são fixos após o ajuste**, e **os pesos são recalculados por amostra**. [Inferência no código](../src/pso_adaptive_fusion.py#L191)

## Equações da fusão

Para o especialista $i$ e a classe $c$, $p_{ic}(x)$ é a probabilidade prevista. A confiança é um menos a entropia normalizada.

$$
h_i(x) = \operatorname{clip}_{[0,1]}\left(1 +
\frac{\sum_{c=1}^{C} p_{ic}(x)\log p_{ic}(x)}{\log C}\right).
$$

O código usa logaritmo natural e considera $0\log 0=0$. Uma distribuição uniforme dá confiança zero. Uma distribuição concentrada em uma classe dá confiança um. Essa confiança descreve a concentração da distribuição do especialista.

$$
z_i(x) = a_i + b_i h_i(x), \qquad
w_i(x) = \frac{e^{z_i(x)}}{\sum_{j=1}^{3} e^{z_j(x)}}, \qquad
p_{\mathrm{final},c}(x) = \sum_{i=1}^{3} w_i(x)p_{ic}(x).
$$

$a_i$ é o termo base do especialista e $b_i$ controla a influência da confiança. O softmax normaliza os pesos **entre os três especialistas**, com soma um. Cada peso multiplica todas as probabilidades do respectivo especialista. O código subtrai o maior logit antes da exponenciação para estabilidade numérica. [Confiança, pesos e média ponderada](../src/pso_adaptive_fusion.py#L23)

## O que o PSO aprende

A posição de cada partícula é $\theta=(a_1,a_2,a_3,b_1,b_2,b_3)$, na ordem SVM, Random Forest e k-NN. O código usa os limites $a_i \in [-4,4]$ e $b_i \in [0,4]$. A configuração padrão usa **24 partículas, 30 iterações e semente 42**. Uma partícula começa em zero, o que dá peso $1/3$ a cada especialista.

A pontuação compara primeiro o F1 Macro OOF, que é a média do F1 das $C$ classes. Entre partículas com o mesmo F1, compara $-\frac{1}{6}\sum_{d=1}^{6}\theta_d^2$. A cada iteração, o enxame usa as melhores posições pessoais e a melhor posição global para atualizar as partículas. O melhor candidato encontrado fornece os coeficientes da fusão. [Ciclo da busca](../src/pso_adaptive_fusion.py#L78) · [Parâmetros padrão](../src/pso_adaptive_fusion.py#L121)

## Diferença em relação ao Stacking

Os dois híbridos usam os mesmos tipos de especialistas e probabilidades OOF. O **Stacking** concatena as probabilidades para treinar uma MLP com uma camada oculta de 64 unidades. A **fusão adaptativa com PSO** aprende seis coeficientes e combina as distribuições por meio de uma média ponderada que varia com a amostra. A contribuição técnica do projeto está nessa regra de combinação. [Configuração do Stacking](../src/02_run_benchmark.py#L98)

## Como ler a figura na apresentação

- A **faixa de treinamento** mostra treino externo, validação interna, probabilidades OOF e ciclo PSO. A função objetivo recebe as classes previstas pela fusão candidata e os rótulos reais do treino.
- A **faixa de inferência** mostra nova amostra, probabilidades dos especialistas, confiança, pesos adaptativos, média ponderada e classe prevista.
- Na **ligação entre as faixas**, os coeficientes aprendidos saem do PSO e entram no cálculo dos pesos. Os especialistas da inferência foram ajustados com o treino externo completo.
