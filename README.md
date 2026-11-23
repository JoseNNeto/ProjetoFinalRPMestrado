# Projeto de Reconhecimento de Padrões — Benchmark de Modelos e Arquitetura Híbrida

Este repositório contém o protocolo experimental, o pipeline de pré-processamento, os algoritmos de classificação (incluindo o sistema híbrido proposto) e os scripts de validação estatística desenvolvidos para a disciplina de **Reconhecimento de Padrões** (Mestrado em Engenharia da Computação).

[Guia do modelo híbrido com PSO](docs/README_MODELO_HIBRIDO.md) · [Perguntas para a apresentação](docs/README_PERGUNTAS_PROFESSOR.md)

---

## 📋 Requisitos do Projeto (Classroom)

- **Equipe:** Até 3 integrantes.
- **Bases de Dados:** 10 datasets distintos (padronizados em Classificação Supervisionada Tabular).
- **Modelos:** 5 classificadores de famílias distintas, sendo **pelo menos 1 modelo híbrido**.
- **Validação Estatística:** Teste de hipótese não-paramétrico de **Friedman** seguido pelo pós-teste de **Nemenyi** (Diagrama de Diferença Crítica).
- **Entregável:** Apresentação oral com slides contendo obrigatoriamente a **figura da arquitetura do sistema híbrido** (desenhada manualmente em ferramentas como Draw.io, Excalidraw ou Figma).

---

## 🗂️ Fase 1: As 10 Bases de Dados (Kaggle)

Para garantir diversidade de domínios (finanças, saúde, sinais biomédicos, indústria, energia, astronomia e telecomunicações) e viabilidade computacional para modelos baseados em kernel ($O(n^2)$ a $O(n^3)$), todas as bases foram padronizadas para **classificação tabular**. Bases originalmente muito extensas passam por **subamostragem estratificada (`stratify=y`) para 20.000 instâncias** antes da validação cruzada.

> **Armazenamento dos Datasets (`dataset/`):** Os arquivos estão versionados compactados em `.csv.gz` para respeitar os limites do GitHub sem necessidade de Git LFS. O `pandas.read_csv("dataset/<arquivo>.csv.gz")` realiza a leitura diretamente, sem necessidade de descompactação manual (ou descompacte com `gzip -d` / 7-Zip se precisar dos `.csv` brutos).

| # | Base de Dados | Domínio | Instâncias Usadas | Tipo de Problema | Coluna Alvo (`target`) | Observações de Limpeza |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **01** | Card Transdata | Finanças / Fraude | 20.000 *(amostra)* | Binária | `fraud` | Subamostragem estratificada (original: 1M) |
| **02** | Smoking & Drinking | Saúde / Biometria | 20.000 *(amostra)* | Binária / Multi | `DRK_YN` / `SMK_stat_type_cd` | Subamostragem estratificada (original: ~990k) |
| **03** | Flight Delay Prediction | Aviação / Logística | 20.000 *(amostra)* | Binária | `ArrDel15` / `Delayed` | Remover colunas de data exata; subamostrar |
| **04** | Dry Bean | Agricultura / Morfologia | 13.543 | Multiclasse (7) | `Class` | Remover 68 linhas duplicadas das 13.611 originais |
| **05** | NASA Nearest Earth Objects | Astrofísica / Órbita | 20.000 *(amostra)* | Binária | `is_hazardous` | Remover IDs/nomes; subamostrar (original: ~338k) |
| **06** | Sloan Digital Sky Survey DR16 | Astronomia | 20.000 *(amostra)* | Multiclasse (3) | `class` | Remover IDs de placa/fibra; subamostrar (70k) |
| **07** | AI4I 2020 Maintenance | Indústria / Sensores | 10.000 | Binária | `Machine failure` | **Dropar** `UDI`, `Product ID` e `TWF, HDF, PWF, OSF, RNF` |
| **08** | Electrical Grid Stability | Energia / Sistemas | 10.000 | Binária | `stabf` | **Dropar** a coluna contínua `stab` (evitar leakage) |
| **09** | Gender Recognition by Voice | Acústica / Sinais | 3.168 | Binária | `label` | Base completa |
| **10** | Telco Customer Churn | Negócios / Telecom | 7.043 | Binária | `Churn` | **Dropar** `customerID`; converter `TotalCharges` para float |

A base 04 utiliza `dataset/dry_bean.csv.gz`, cópia do ARFF oficial convertida para CSV. A UCI disponibiliza a [Dry Bean](https://doi.org/10.24432/C50S4B) sob licença CC BY 4.0. A preparação grava os novos metadados em `results/dataset_metadata_dry_bean.csv`.

---

## ⚙️ Fase 2: Pipeline de Pré-processamento Universal

Para evitar **vazamento de dados (*data leakage*)**, todas as transformações são encapsuladas dentro de um `Pipeline` do `scikit-learn` e ajustadas (`fit`) **exclusivamente nos dados de treino de cada dobra (fold)**:

1. **Codificação do Alvo (`y`):** `LabelEncoder` para mapear as classes em inteiros de $0$ a $C-1$.
2. **Atributos Numéricos (`int64`, `float64`):**
   - **Imputação:** `SimpleImputer(strategy='median')` (robusto a *outliers*).
   - **Escalonamento:** `StandardScaler()` ($\mu = 0, \sigma = 1$), essencial para classificadores baseados em distância e margem (SVM, KNN e MLP).
3. **Atributos Categóricos (`object`, `category`, `bool`):**
   - **Imputação:** `SimpleImputer(strategy='most_frequent')`.
   - **Codificação:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

---

## 🧠 Fase 3: Os 6 Modelos e as Arquiteturas Híbridas

Os 6 modelos cobrem diferentes paradigmas matemáticos de Reconhecimento de Padrões:

1. **SVM (`SVC(kernel='rbf', probability=True)`):** Classificador baseado em margem máxima e projeção não-linear via kernel radial.
2. **KNN (`KNeighborsClassifier(n_neighbors=5)`):** Classificador não-paramétrico baseado em distância euclidiana no espaço de atributos.
3. **Random Forest (`RandomForestClassifier(n_estimators=100)`):** Comitê (*ensemble*) homogêneo baseado em árvores de decisão e *bagging*.
4. **MLP (`MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True)`):** Rede neural conexionista de múltiplas camadas.
5. **Modelo Híbrido Proposto — Stacking Heterogêneo com Meta-Classificador Neural:**
   - **Nível 0 (Especialistas de Famílias Distintas):** **SVM (RBF)**, **Random Forest** e **KNN** recebem os dados brutos em pipelines próprios. Nos três folds da validação cruzada interna (`cv=3`), cada pipeline ajusta a imputação, a padronização e a codificação categórica usando apenas o treino do fold. Em seguida, gera as probabilidades ($P_1, P_2, P_3$) para as amostras de validação que alimentarão o meta-classificador.
   - **Concatenação de Meta-Atributos:** As probabilidades preditas pelos três modelos base são concatenadas formando um novo espaço de representação $Z \in \mathbb{R}^{3 \times C}$.
   - **Nível 1 (Meta-Learner):** Uma rede neural **MLP (`hidden_layer_sizes=(64,)`)** recebe o vetor $Z$ e aprende a fronteira de decisão final, ponderando dinamicamente qual especialista é mais confiável em cada região do espaço.
6. **Híbrido (PSO Adaptativo):** Combina probabilidades de **SVM (RBF)**, **Random Forest** e **KNN** usando pesos por instância baseados na confiança de cada especialista. O PSO aprende os coeficientes a partir das probabilidades out-of-fold do treino externo.

> **Nota para a Figura da Arquitetura (Slide 3):** Desenhe o fluxo da esquerda para a direita: `Dados Brutos` $\rightarrow$ `ColumnTransformer (Imputer + Scaler / OneHot)` $\rightarrow$ `Split Paralelo Nível 0 [SVM | Random Forest | KNN]` $\rightarrow$ `Vetor de Probabilidades Concatenado (Z)` $\rightarrow$ `Meta-Classificador Nível 1 (MLP)` $\rightarrow$ `Classe Predita`.

---

## 🧪 Fase 4: Protocolo de Treinamento e Avaliação Estatística

### 1. Validação Cruzada
- **Estratégia:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` (mantém a proporção original das classes em todas as dobras).
- **Total de Execuções:** $10 \text{ bases} \times 6 \text{ modelos} \times 5 \text{ folds} = 300 \text{ treinamentos}$.
- **Métricas Coletadas:**
  - **F1-Score Macro** (`f1_macro`) — *Métrica principal para o ranking estatístico, pois penaliza erros nas classes minoritárias.*
  - **Acurácia Balanceada** (`balanced_accuracy`).
  - **Tempo Médio de Treinamento** (`fit_time`).

As métricas do híbrido da execução original, preservadas em `results/metrics_raw.csv`, foram geradas antes da correção do pré-processamento na validação cruzada interna.

Para executar apenas o híbrido PSO em uma base preparada, com dois folds:

```powershell
python src/02_run_benchmark.py --datasets 07_ai4i_2020 --models pso --n-splits 2 --results-path results/pso_smoke.csv
```

Para avaliar os seis modelos em todas as bases preparadas, com cinco folds:

```powershell
python src/02_run_benchmark.py --results-path results/metrics_raw_dry_bean.csv
```

O benchmark preserva CSVs existentes. Inicie a avaliação com Dry Bean em um novo CSV. Use `--resume` com o mesmo caminho e número de folds para completar as combinações faltantes.

Com os 300 registros do benchmark concluídos, gere a análise em um diretório separado.

```powershell
python src/03_stats_analysis.py
```

A análise lê `results/metrics_raw_dry_bean.csv` e grava em `results/stats_dry_bean/`. O script recusa grades incompletas e preserva arquivos de saída existentes.

### 2. Teste de Hipótese (Friedman-Nemenyi)
Como os desempenhos nas 10 bases não seguem distribuição normal nem possuem variâncias homogêneas (violando premissas da ANOVA paramétrica), utiliza-se o protocolo não-paramétrico de Demšar (2006):

1. **Cálculo dos Rankings:** Em cada base de dados, os 6 modelos recebem posições de `1` (melhor F1-Score Macro) a `6` (pior F1-Score Macro).
2. **Teste de Friedman (`scipy.stats.friedmanchisquare`):** Testa a hipótese nula ($H_0$) de que todos os 6 modelos possuem o mesmo desempenho médio. Se $p < 0.05$, rejeita-se $H_0$.
3. **Pós-teste de Nemenyi (`scikit-posthocs` / `autorank`):** Calcula a **Diferença Crítica (CD)** para nível de significância $\alpha = 0.05$ e gera o **Diagrama de Diferença Crítica (CD Diagram)**, conectando por barras horizontais os modelos que não apresentam diferença estatisticamente significativa entre si.

---

## 💻 Fase 5: Estrutura Sugerida do Repositório

```text
├── data/                   # CSVs brutos ou script de download via API do Kaggle
├── src/
│   ├── 01_prepare_data.py  # Limpeza específica, subamostragem estratificada e tabela de metadados
│   ├── 02_run_benchmark.py # Pipeline universal + 5-Fold Stratified CV dos 6 modelos
│   └── 03_stats_analysis.py# Teste de Friedman, pós-teste de Nemenyi e plot do CD Diagram
├── results/
│   ├── metrics_raw.csv     # Resultados históricos por fold
│   ├── metrics_raw_dry_bean.csv # Resultados do benchmark com Dry Bean
│   ├── summary_table.csv   # Média ± Desvio Padrão e Rank Médio por base
│   └── cd_diagram.png      # Gráfico de Diferença Crítica para os slides
├── docs/
│   └── arquitetura_hibrido.drawio # Arquivo editável da figura exigida na apresentação
└── README.md
```

---

## 🎤 Roteiro Objetivo para os Slides da Apresentação

1. **Slide 1 — Contextualização e Bases de Dados:** Tabela resumo das 10 bases (domínio, nº de instâncias, atributos, classes e razão de desbalanceamento).
2. **Slide 2 — Metodologia e Pré-processamento:** Fluxo do `ColumnTransformer` dentro do `Pipeline` (prevenção de *data leakage*) e validação cruzada `5-Fold Stratified`.
3. **Slide 3 — Arquitetura do Sistema Híbrido:** Diagrama em blocos do *Stacking Heterogêneo* (Nível 0 + Concatenação + Meta-Learner Nível 1) e justificativa teórica.
4. **Slide 4 — Resultados Experimentais:** Tabela comparativa (F1-Score Macro: Média $\pm$ DP) com destaque em negrito para o melhor modelo por base e linha final de **Rank Médio**.
5. **Slide 5 — Análise Estatística (Friedman-Nemenyi):** Estatística $\chi^2_F$, $p\text{-valor}$ do teste de Friedman e o **Diagrama de Diferença Crítica (CD)** discutindo a superioridade ou equivalência estatística do modelo híbrido.
6. **Slide 6 — Conclusões e Limitações:** Trade-off entre ganho de desempenho e custo computacional (`fit_time`) do modelo híbrido frente aos classificadores individuais.