# Perguntas para a apresentação

Respostas curtas sobre a arquitetura, o protocolo e os resultados finais do projeto. Para preparar a explicação da figura, veja o [guia do modelo híbrido](README_MODELO_HIBRIDO.md).

## Arquitetura e treinamento

### 1. O que o projeto propõe?

Uma forma de combinar três classificadores tabulares. O PSO aprende como a importância de cada especialista e a confiança da sua previsão influenciam o peso usado na fusão.

### 2. O que torna o modelo híbrido?

Ele reúne SVM RBF, Random Forest e k-NN, que têm formas diferentes de aprender. A camada de fusão combina as probabilidades dos três.

### 3. O PSO treina a SVM, a floresta ou o k-NN?

O PSO otimiza **seis coeficientes da fusão**. Os especialistas são ajustados pelos seus algoritmos usuais, e as probabilidades OOF são reutilizadas para avaliar todas as partículas.

### 4. Por que a fusão é adaptativa?

Os coeficientes ficam fixos após o treino. Na inferência, a confiança de cada especialista é calculada a partir da distribuição que ele prevê para aquela amostra. O softmax usa essas confianças e os coeficientes para gerar três pesos por amostra.

### 5. Confiança alta significa que a previsão está correta?

Confiança alta significa que o especialista concentrou sua probabilidade em poucas classes. Uma distribuição concentrada ainda pode apontar para a classe errada. O PSO usa os rótulos de treino e as predições OOF para aprender a contribuição desse sinal.

### 6. Qual é a função objetivo?

Maximizar o **F1 Macro das predições OOF fundidas**. O F1 é calculado sobre as amostras OOF reunidas. Se duas partículas obtêm o mesmo F1, o desempate favorece a menor média dos quadrados dos coeficientes.

### 7. Qual é a diferença para o Stacking MLP?

O Stacking concatena as probabilidades OOF e ajusta uma MLP como meta-classificador. O PSO aprende coeficientes para uma média ponderada adaptativa. Na apresentação, destaque que os dois usam os mesmos tipos de especialistas e que a contribuição proposta está na regra de fusão. [Código dos dois híbridos](../src/02_run_benchmark.py#L98)

### 8. O PSO roda de novo para classificar uma amostra?

A busca ocorre no treinamento. Para classificar, o modelo gera as probabilidades com os pipelines já ajustados, calcula as confianças e usa os coeficientes aprendidos para obter os pesos, a distribuição final e a classe de maior probabilidade.

## Protocolo experimental

### 9. Por que cinco folds externos e três internos?

Os cinco externos medem o desempenho em amostras reservadas para teste. Os três internos geram as probabilidades OOF usadas para ajustar a fusão. Os seis modelos usam as mesmas divisões externas em cada base. São **10 bases × 6 modelos × 5 folds = 300 registros**. [Benchmark](../src/02_run_benchmark.py#L139)

### 10. Como vocês evitaram vazamento no pré-processamento?

No PSO e no Stacking, o pré-processamento faz parte do pipeline de cada especialista. Em cada fold interno, o pipeline aprende imputação, escala e categorias apenas no treino interno. A validação interna é transformada com esse ajuste, e o teste externo fica reservado à avaliação. [Teste de isolamento do PSO](../tests/test_pso_adaptive_fusion.py#L97) · [Teste de isolamento do Stacking](../tests/test_benchmark.py#L70)

### 11. Por que usar F1 Macro e acurácia balanceada?

O F1 Macro calcula o F1 de cada classe a partir da precisão e do recall e dá o mesmo peso a todas as classes. A acurácia balanceada é a média do recall por classe. As duas métricas ajudam a enxergar o desempenho nas classes minoritárias. O F1 Macro é a métrica principal do experimento.

### 12. Por que trocar a EMG pela Dry Bean?

Na EMG, os mesmos participantes apareciam nos conjuntos de treino e teste, e as leituras tinham dependência temporal. A Dry Bean mantém um problema multiclasse tabular compatível com o protocolo estratificado do benchmark. Após retirar 68 duplicatas exatas, ficaram **13.543 linhas, 16 atributos numéricos e sete classes**. [Preparação da base](../src/01_prepare_data.py#L115) · [Metadados das bases](../results/dataset_metadata_dry_bean.csv)

### 13. O que a ablação mostrou?

O piloto comparou fusão uniforme, PSO com pesos fixos, PSO adaptativo e Stacking em amostras estratificadas de 600 linhas de AI4I e Steel Plates, com dois folds externos. No AI4I, as duas fusões PSO empataram em F1 Macro médio de **0,6923**. No Steel Plates, o adaptativo obteve **0,7023**, ante **0,6724** do PSO fixo e **0,6963** da fusão uniforme. O piloto dá evidência exploratória sobre a adaptação por confiança. [Resultados da ablação](../results/hybrid_ablation_pilot.csv)

## Resultados finais

### 14. Qual modelo ficou em primeiro lugar?

O PSO Adaptativo obteve o **melhor rank médio, 2,20**, seguido por MLP e Random Forest, ambas com 2,80. Pela média aritmética de F1 Macro das dez bases, a Random Forest ficou à frente, com **0,8102**, e o PSO obteve **0,8044**. O rank responde à posição relativa em cada base. A média aritmética leva em conta a magnitude dos escores. [Métricas por fold](../results/metrics_raw_dry_bean.csv) · [Tabela por base e modelo](../results/stats_dry_bean/summary_table_formatted.csv)

### 15. O Friedman mostra que o PSO é superior aos outros modelos?

O teste global de Friedman obteve $p \approx 0{,}0100$ e indicou diferença de desempenho entre os modelos. A comparação por pares vem do Nemenyi. A 5%, a matriz registrou diferença significativa apenas entre **PSO Adaptativo e KNN**, com $p \approx 0{,}03345$, e o PSO teve o melhor rank desse par. Para PSO e Stacking, $p \approx 0{,}2060$, o que indica que o experimento ficou sem evidência suficiente de diferença entre eles a 5%. [Resumo estatístico](../results/stats_dry_bean/stats_summary.txt) · [Matriz de p-valores](../results/stats_dry_bean/nemenyi_pvalues.csv)

### 16. Qual é a unidade usada no teste estatístico?

Cada base contribui como um bloco. Primeiro calculamos o F1 Macro médio dos cinco folds para cada par base e modelo. Essa tabela de **dez bases por seis modelos** alimenta Friedman e Nemenyi. Os folds servem para estimar o desempenho dentro da base. [Agregação no código](../src/03_stats_analysis.py#L76)

### 17. Qual foi o custo computacional do híbrido?

O PSO somou cerca de **51 min 41 s de ajuste** nos 50 registros do modelo. O Stacking somou cerca de **33 min 52 s**. Cada híbrido treina três especialistas nos folds internos e depois ajusta três pipelines finais. No piloto, gerar as probabilidades OOF levou mais tempo que buscar os pesos. O `fit_time` do benchmark termina antes da predição no teste externo. [Tempos do benchmark](../results/metrics_raw_dry_bean.csv) · [Tempos do piloto](../results/hybrid_ablation_pilot.csv)

### 18. Quais limitações vale citar?

O estudo cobre dez bases tabulares e uma configuração de PSO. A entropia mede concentração, e o F1 OOF é utilizado para escolher os pesos durante o treino. O teste externo é a referência para generalização dentro do protocolo. A ablação piloto cobre apenas duas bases, e a diferença estatística detectada no benchmark vale para o par identificado pelo Nemenyi.

## Resposta de fechamento

> O híbrido aprende como combinar três especialistas de acordo com a confiança de cada previsão. No benchmark, teve o melhor rank médio, com custo de treinamento maior. O Nemenyi detectou diferença significativa em relação ao KNN, e as outras comparações ficaram acima do limiar de 5%.
