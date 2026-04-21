# Previsão do Nível de Obesidade usando Regressão Logística

## Visão Geral

Este projeto aplica técnicas de aprendizado de máquina para prever os níveis de obesidade com base no estilo de vida e em atributos físicos. O conjunto de dados é pré-processado utilizando métodos de normalização e codificação, e duas estratégias de classificação multiclasse são avaliadas:

* Um-contra-Todos (OvR - *One-vs-Rest*)
* Um-contra-Um (OvO - *One-vs-One*)

Ambas as abordagens utilizam a Regressão Logística como classificador base.

---

## Conjunto de Dados

O conjunto de dados é carregado diretamente de uma fonte online e contém:

* Variáveis numéricas (ex: idade, altura, peso)
* Variáveis categóricas (ex: hábitos alimentares, atividade física)
* Variável alvo: `NObeyesdad` (classificação do nível de obesidade)

---

## Pré-processamento de Dados

### 1. Tratamento de Variáveis Numéricas

* As colunas numéricas são identificadas usando o tipo de dado `float64`.
* A padronização é aplicada usando o `StandardScaler`:
  * Média = 0
  * Desvio padrão = 1

### 2. Tratamento de Variáveis Categóricas

* As variáveis categóricas são identificadas usando o tipo de dado `object`.
* A coluna alvo (`NObeyesdad`) é excluída da codificação.
* A codificação *One-Hot* é aplicada usando o `OneHotEncoder`:
  * `drop='first'` evita a multicolinearidade.

### 3. Codificação da Variável Alvo

* A variável alvo é convertida em categorias numéricas usando `.cat.codes`.

---

## Separação entre *Features* e Variável Alvo

* *Features* (`X`): Todas as colunas, exceto `NObeyesdad`
* Alvo (`y`): Níveis de obesidade codificados

---

## Divisão em Treino e Teste

* Os dados são divididos em conjuntos de treinamento e teste:
  * Conjunto de treinamento: 67%
  * Conjunto de teste: 33%
* `random_state=42` garante a reprodutibilidade.

---

## Modelos

### 1. Um-contra-Todos (OvR - *One-vs-Rest*)

* Utiliza `LogisticRegression` com:
  * `multi_class='ovr'`
  * `max_iter=1000`
* Treina um classificador por classe contra todas as outras.

### 2. Um-contra-Um (OvO - *One-vs-One*)

* Utiliza `OneVsOneClassifier` com Regressão Logística.
* Treina um classificador para cada par de classes.

---

## Avaliação

* As previsões são feitas no conjunto de teste.
* O desempenho é medido usando a métrica de acurácia (*accuracy score*).

```python
print(accuracy_score(ytest, y_pred))     # Acurácia OvR
print(accuracy_score(ytest, yovo_pred))  # Acurácia OvO
```

---

## Bibliotecas Utilizadas

```bash
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
```

---

## Conceitos-Chave

* Normalização de dados
* Codificação *One-hot*
* Estratégias de classificação multiclasse
* Regressão logística
* Avaliação de modelo usando acurácia

---

## Possíveis Melhorias

* Ajuste de hiperparâmetros (ex: força de regularização)
* Validação cruzada (*Cross-validation*)
* Testar outros classificadores (Random Forest, SVM, Gradient Boosting)
* Seleção de *features* ou redução de dimensionalidade
* Tratamento de desbalanceamento de classes, se presente
