# Guia de Preparação — Reunião de Acompanhamento (Reconhecimento de Padrões)

Este documento reúne tudo o que você precisa para explicar com segurança ao professor **o que está sendo feito**, **por que cada decisão técnica foi tomada**, **se o projeto está no caminho certo** e **quais pontos vale alinhar com ele amanhã**.

---

## 1. Resumo Executivo: Estamos fazendo o certo?

**Sim.** A arquitetura experimental já implementada no repositório atende integralmente às boas práticas de Aprendizado de Máquina e aos requisitos do projeto:

| Requisito da Disciplina | O que foi projetado / implementado | Status |
| :--- | :--- | :--- |
| **10 bases tabulares diversas** | 10 datasets de domínios distintos (finanças, saúde, aviação, sinais EMG, astronomia, indústria, energia, metalurgia, telecom) | Arquivos prontos e limpos mapeados em `src/01_prepare_data.py` |
| **5 classificadores distintos** | `SVM (RBF)`, `KNN`, `Random Forest`, `MLP` e **Modelo Híbrido (Stacking Heterogêneo)** | Implementados em `src/02_run_benchmark.py` |
| **Prevenção de *data leakage*** | Remoção prévia de colunas com vazamento + `ColumnTransformer` estritamente dentro do `Pipeline` por dobra | Implementado em `01_prepare_data.py` e `02_run_benchmark.py` |
| **Validação cruzada rigorosa** | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` (250 execuções no total) | Implementado em `src/02_run_benchmark.py` |
| **Avaliação estatística** | Teste de Friedman + pós-teste de Nemenyi + Diagrama de Diferença Crítica (CD Diagram, $\alpha = 0,05$) | Implementado em `src/03_stats_analysis.py` |
| **Execução e entregáveis visuais** | Rodar os scripts na `.venv`, desenhar a figura da arquitetura híbrida manualmente e montar os 6 slides | **Próxima etapa** (após a reunião / validação) |

> **Frase-chave para abrir a reunião:**  
> *"Professor, o protocolo experimental completo já está estruturado e codificado no repositório em três etapas reprodutíveis: preparação e saneamento das 10 bases, benchmark com 5-Fold Stratified CV encapsulado em Pipeline para evitar vazamento de dados, e análise estatística não-paramétrica de Friedman-Nemenyi. Antes de disparar as 250 execuções completas e montar os slides, queria alinhar algumas decisões de limpeza das bases e validar se a arquitetura do híbrido está alinhada com o que o senhor espera."*

---

## 2. O que está acontecendo no código (Explicação Passo a Passo)

### Etapa 1 — Saneamento e Padronização das Bases (`src/01_prepare_data.py`)

O objetivo aqui é transformar 10 arquivos heterogêneos (alguns com milhões de linhas ou dezenas de colunas problemáticas) em bases tabulares limpas e padronizadas em `data/processed/`.

#### A) Por que subamostrar as 6 bases gigantes para 20.000 instâncias?
- Bases como **EMG** (~4,24 milhões de linhas), **Card Transdata** (~1 milhão), **Smoking & Drinking** (~991 mil), **Flight Delay** (~563 mil), **NASA NEO** (~338 mil) e **SDSS DR16** (70 mil) tornariam o treinamento inviável.
- **Justificativa matemática:** O `SVC` com kernel RBF (`probability=True`) tem complexidade computacional entre $O(n^2 \cdot d)$ e $O(n^3)$ e ainda roda calibração de probabilidades interna (Platt scaling com validação cruzada). Com 1 milhão de linhas, o `SVM` e o `Stacking` levariam dias ou esgotariam a memória RAM.
- **Como é feito:** Usamos **subamostragem estratificada (`stratify=target`, `random_state=42`)**, preservando exatamente a proporção original das classes.

#### B) Limpeza específica para evitar *Data Leakage* e ruído
Se o professor perguntar *"como vocês garantiram que não entrou variável resposta disfarçada nos dados?"*, você tem respostas concretas para cada base crítica:

1. **AI4I 2020 Maintenance (10.000 linhas):** Removemos `UDI`, `Product ID` (identificadores) e `TWF, HDF, PWF, OSF, RNF` (os 5 modos específicos de falha, pois o alvo `Machine failure` é derivado diretamente deles).
2. **Electrical Grid Stability (10.000 linhas):** Removemos `stab` (variável contínua cujo sinal positivo/negativo define exatamente a classe `stabf`).
3. **Flight Delay Prediction (20.000 de ~527 mil voos válidos):**
   - Filtramos linhas em que `ArrDel15` é `NaN` (36.843 voos cancelados/desviados sem atraso de chegada registrado).
   - Removemos **todas as colunas pós-voo** (ex.: `ArrDelay`, `DepDelay`, `DepTime`, `ArrTime`, `TaxiOut`, `TaxiIn`, `AirTime`, `CarrierDelay`, `Cancelled`, `Diverted`, etc.) — só conhecemos esses valores *depois* que o voo já aconteceu.
   - Removemos a data exata (`FlightDate`), códigos de aeronave (`Tail_Number`) e colunas vazias (`Unnamed: 119`), mantendo apenas atributos **estritamente pré-voo** (`Quarter`, `Month`, `DayofMonth`, `DayOfWeek`, `Operating_Airline`, `OriginState`, `DestState`, `CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`, `Distance`, `DistanceGroup`).
4. **NASA NEO (20.000 de 338.199):** Removemos `neo_id`, `name` e `orbiting_body` (que tem valor único `'Earth'` em 100% da base).
5. **SDSS DR16 (20.000 de 70.000):** Removemos identificadores astronômicos/instrumentais (`objid`, `run`, `rerun`, `camcol`, `field`, `plate`, `mjd`, `fiberid`).
6. **Telco Customer Churn (7.043):** Removemos `customerID` e convertemos `TotalCharges` para numérico (`errors='coerce'`), deixando eventuais `NaN` para serem imputados **apenas dentro de cada fold** no `Pipeline`.
7. **EMG Gesture Recognition (20.000 de 4.237.907):** Removemos `time` e `label` (ID do sujeito, 36 valores) e mantivemos os 8 canais do sensor (`channel1` a `channel8`) para predizer `class` (8 classes, 0 a 7).
8. **Steel Plates Faults (1.941):** Removemos `id` e convertemos as 7 colunas *one-hot* mutuamente exclusivas (`Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, `Other_Faults`) em um único alvo multiclasse (`target`).

---

### Etapa 2 — Pipeline Universal e Modelo Híbrido (`src/02_run_benchmark.py`)

#### A) Como garantimos Zero *Data Leakage* no Pré-processamento
Este é um dos pontos que professores de Pós-Graduação mais valorizam:
- Nunca calculamos mediana, média, desvio padrão ou categorias no dataset inteiro antes da divisão.
- Usamos `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` (80% treino, 20% teste em cada dobra).
- Em **cada uma das 5 dobras**, o `ColumnTransformer` é encapsulado dentro de um `Pipeline([('preprocessor', ...), ('classifier', ...)])`:
  - **Atributos Numéricos:** `SimpleImputer(strategy='median')` (robusto a *outliers*) $\rightarrow$ `StandardScaler()` ($\mu=0, \sigma=1$, essencial para `SVM`, `KNN` e `MLP`).
  - **Atributos Categóricos:** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- O método `.fit()` do `Pipeline` enxerga **apenas os 80% de treino**; os 20% de teste passam somente por `.transform()` / `.predict()`.

#### B) Os 5 Modelos e a Arquitetura do Sistema Híbrido
Os 4 classificadores base cobrem paradigmas distintos de Reconhecimento de Padrões:
1. **SVM (`SVC(kernel='rbf', probability=True)`):** fronteira de margem máxima não-linear via truque do kernel.
2. **KNN (`KNeighborsClassifier(n_neighbors=5)`):** aprendizado baseado em instâncias / distância euclidiana local.
3. **Random Forest (`RandomForestClassifier(n_estimators=100)`):** comitê (*ensemble*) homogêneo baseado em árvores de decisão e *bagging*.
4. **MLP (`MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True)`):** rede neural conexionista multicamadas.

O **5º modelo é o Híbrido Proposto (*Stacking* Heterogêneo com Meta-Classificador Neural)**:
- **Nível 0 (Especialistas):** `SVM (RBF)` + `Random Forest` + `KNN` são treinados em paralelo usando **validação cruzada interna (`cv=3`)** sobre a dobra de treino para gerar previsões *out-of-fold* de probabilidades a posteriori ($P_1, P_2, P_3$). O `cv=3` interno evita que os modelos de Nível 0 memorizem o próprio treino ao alimentar o Nível 1.
- **Espaço de Meta-Atributos ($Z$):** As probabilidades dos 3 especialistas são concatenadas formando um novo vetor de características de entrada para o meta-aprendiz.
- **Nível 1 (Meta-Learner):** Uma `MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True)` recebe $Z$ e aprende **não-linearmente** como ponderar cada especialista em diferentes regiões do espaço de atributos (indo além de uma combinação linear simples como Regressão Logística).

---

### Etapa 3 — Métricas e Análise Estatística (`src/03_stats_analysis.py`)

#### A) Por que `F1-Score Macro` e `Balanced Accuracy` (e não Acurácia simples)?
Várias das 10 bases são fortemente desbalanceadas (ex.: fraude em cartão de crédito, objetos espaciais perigosos, falhas industriais). A acurácia comum pode dar 98% simplesmente chutando a classe majoritária. Já o **F1-Score Macro** e a **Acurácia Balanceada** dão peso igual a todas as classes, penalizando modelos que ignoram a classe minoritária. Além disso, registramos o `fit_time` para analisar o *trade-off* entre desempenho e custo computacional.

#### B) Por que Teste de Friedman + Pós-teste de Nemenyi?
Conforme a literatura clássica de avaliação de classificadores em múltiplas bases (**Demšar, 2006**):
- Os desempenhos em 10 bases de domínios distintos **não** satisfazem normalidade nem homocedasticidade (variâncias iguais), invalidando a ANOVA paramétrica.
- **Passo 1 (Rankings):** Em cada base, os 5 modelos recebem posições de `1` (melhor F1-Score Macro) a `5` (pior).
- **Passo 2 (Friedman):** Verifica a hipótese nula ($H_0$) de que todos os modelos têm o mesmo rank médio. Se $p < 0,05$, rejeita-se $H_0$.
- **Passo 3 (Pós-teste de Nemenyi):** Calcula a **Diferença Crítica ($CD$)** para $\alpha = 0,05, k = 5, N = 10$ ($CD = 1,9288$) e gera o **Diagrama de Diferença Crítica (`results/cd_diagram.png`)**. Pares de modelos cuja distância entre ranks médios for menor que a $CD$ são conectados por barras horizontais (indicando equivalência estatística).

> **Importante para a sua tranquilidade científica:** O experimento é igualmente válido quer o modelo híbrido fique isolado em 1º lugar, quer fique estatisticamente empatado com `Random Forest` ou `SVM` no Diagrama de Diferença Crítica. O papel da disciplina é **aplicar o protocolo corretamente e analisar criticamente o ganho vs. custo (`fit_time`)**.

---

## 3. Pontos Estratégicos para Confirmar com o Professor Amanhã

Leve estes **4 pontos objetivos** para mostrar maturidade técnica e validar qualquer detalhe antes de rodar o benchmark completo:

1. **Base #09 (`Steel Plates Faults` vs `Gender Recognition by Voice`):**
   - *Contexto:* O `README.md` original citava *Gender Recognition by Voice*, mas o dataset real colocado na pasta é **Steel Plates Faults** (1.941 amostras, 7 tipos exclusivos de falha em chapas de aço).
   - *Pergunta:* *"Professor, na base 9 acabamos adotando a Steel Plates Faults (multiclasse com 7 classes de defeitos industriais), que aumenta inclusive a variedade de problemas multiclasse do benchmark. Tudo bem manter ela?"*
2. **Base #04 (`EMG` — 4,24M linhas, 36 sujeitos e classe 0 de repouso):**
   - *Contexto:* No arquivo completo de EMG existem ~4,24 milhões de linhas, 8 classes (`0` = não marcado/repouso, `1..7` = gestos) e a coluna `label` (ID dos 36 sujeitos). No momento, removemos `time` e `label` e fizemos subamostragem estratificada para 20.000 linhas mantendo as 8 classes.
   - *Pergunta:* *"Na base de EMG, faz sentido manter a classe 0 (repouso) com subamostragem estratificada global para 20k linhas, ou o senhor recomenda filtrar a classe 0 / agrupar de outra forma?"*
3. **Base #02 (`Smoking & Drinking`):**
   - *Contexto:* A base possui dois possíveis alvos (`DRK_YN` binário para consumo de álcool e `SMK_stat_type_cd` multiclasse para tabagismo). Escolhemos `DRK_YN` como alvo e mantivemos as variáveis clínicas/demográficas (incluindo `SMK_stat_type_cd`) como atributos preditores.
   - *Pergunta:* *"Usar `DRK_YN` como alvo mantendo o status de tabagismo como feature está adequado?"*
4. **Limpeza do `Flight Delay`:**
   - *Contexto:* Removemos todas as colunas pós-decolagem/pós-pouso (atraso na saída, tempo de voo, cancelamento) e mantivemos apenas dados pré-voo (`Month`, `DayOfWeek`, `Operating_Airline`, estados, horários previstos e distância), filtrando voos cancelados (`ArrDel15` nulo).
   - *Pergunta rápida:* Apenas confirmar que essa modelagem estritamente pré-voo atende ao critério de ausência de vazamento.

---

## 4. Possíveis Perguntas Difíceis do Professor (e como responder)

1. **"Por que usar `MLP` como meta-classificador em vez de Regressão Logística no Stacking?"**
   - *Resposta:* Porque queremos que o meta-aprendiz seja capaz de aprender fronteiras de decisão **não-lineares** no espaço das probabilidades preditas pelos especialistas (por exemplo, confiando mais no `SVM` em certa região e mais na `Random Forest` em outra). Além disso, atende à proposta de um sistema híbrido combinando modelos clássicos (margem, distância, árvores) com um meta-classificador neural.
2. **"Qual o risco de colocar o `ColumnTransformer` fora do `StackingClassifier` em relação ao `cv=3` interno?"**
   - *Resposta:* Excelente observação! Hoje o `ColumnTransformer` é ajustado estritamente dentro de cada fold de treino do `StratifiedKFold(5)` externo, portanto **zero informação do conjunto de teste vaza**. Dentro desse fold de treino, os 3 estimadores do `cv=3` interno do Stacking compartilham a mesma matriz já escalonada/imputada. O vazamento entre folds internos de `StandardScaler`/`SimpleImputer` costuma ser desprezível, mas se o senhor achar necessário rigor máximo, podemos até embutir o pré-processador dentro de cada estimador de Nível 0 (apenas observando o impacto no tempo de execução).
3. **"Por que não usaram busca de hiperparâmetros (`GridSearchCV`)?"**
   - *Resposta:* Como o objetivo é comparar 5 famílias distintas em 10 bases diferentes sob um protocolo uniforme ($250$ treinamentos base, mais $3 \times 3$ folds internos no Stacking), fixar hiperparâmetros padrão bem estabelecidos mantém o experimento computacionalmente viável e comparável sem viés de otimização desigual entre modelos. *(Ou, se houver tempo após o benchmark base, pode ser citado como trabalho futuro / limitação no Slide 6.)*
4. **"Como vocês garantiram que o código e os dados sejam reprodutíveis pela equipe e na entrega?"**
   - *Resposta:* Fixamos `random_state=42` em todas as etapas (subamostragem, `StratifiedKFold`, `RandomForest`, `MLP` e `SVC`), comprimimos os datasets em `.csv.gz` (reduzindo de ~756 MB para ~111 MB no GitHub sem depender de Git LFS) e automatizamos todo o fluxo em `requirements.txt` + `src/01` a `src/03`.

---

## 5. O que fazer depois da reunião (Sequência Prática)

Quando você voltar da reunião com o "ok" (ou eventuais ajustes do professor):
1. Adicionar `.venv/` ao `.gitignore`.
2. Instalar as dependências na `.venv`: `pip install -r requirements.txt`.
3. Rodar `python src/01_prepare_data.py` e conferir `results/dataset_metadata.csv`.
4. Testar `python src/02_run_benchmark.py --datasets 09_steel_plates_faults` e depois rodar o benchmark completo (`python src/02_run_benchmark.py`).
5. Rodar `python src/03_stats_analysis.py` para gerar `results/summary_table_formatted.csv`, `results/stats_summary.txt` e `results/cd_diagram.png`.
6. Desenhar a arquitetura manualmente (`docs/arquitetura_hibrido.drawio`), montar os 6 slides conforme o roteiro do `README.md` e fazer o `git commit` + `git push`.
