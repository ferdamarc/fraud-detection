# Detecção de Fraude em Transações Financeiras

> Projeto de portfólio em Data Science — IEEE-CIS Fraud Detection (Kaggle)

## Visão Geral

Este projeto desenvolve um pipeline completo de Machine Learning para detectar transações financeiras fraudulentas, utilizando o dataset da competição [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) do Kaggle.

O desafio central é a **extrema desbalanceamento de classes**: apenas ~3.5% das transações são fraudes, o que torna métricas como acurácia enganosas e exige técnicas especializadas.

## Dataset

- **Fonte:** Kaggle — IEEE-CIS Fraud Detection
- **Tabelas:** `train_transaction.csv` (590k transações) + `train_identity.csv` (141k registros de identidade)
- **Target:** `isFraud` (binário: 0 = legítimo, 1 = fraude)
- **Features:** ~430 variáveis (tempo, valor, dispositivo, rede, comportamento)

> Os dados brutos **não estão incluídos** neste repositório por questões de tamanho (1.3 GB). Veja instruções de download abaixo.

## Estrutura do Projeto

```
fraud_detection/
├── notebooks/
│   ├── 01_data_understanding.ipynb    # Exploração inicial e merge das tabelas
│   ├── 02_eda.ipynb                   # Análise Exploratória de Dados
│   ├── 03_preprocessing.ipynb         # Pipeline de pré-processamento
│   ├── 04_modeling.ipynb              # Treinamento e comparação de modelos
│   └── 05_evaluation.ipynb            # Métricas, SHAP e conclusões
├── src/
│   └── utils.py                       # Funções auxiliares reutilizáveis
├── reports/
│   └── figures/                       # Gráficos gerados pelos notebooks
├── models/                            # Modelos serializados (não versionados)
├── environment.yml                    # Ambiente conda reproduzível
└── README.md
```

## Como Reproduzir

### 1. Clone o repositório

```bash
git clone https://github.com/ferdamarc/fraud-detection.git
cd fraud-detection
```

### 2. Crie o ambiente conda

```bash
conda env create -f environment.yml
conda activate fraud_detection
```

### 3. Baixe os dados

Faça o download do dataset no [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) e coloque os arquivos CSV em uma pasta `data/` acessível. Ajuste o caminho `DATA_DIR` nos notebooks conforme sua estrutura local.

### 4. Execute os notebooks em ordem

```bash
jupyter lab
```

## Metodologia

| Fase | Notebook | Descrição |
|------|----------|-----------|
| 1 | `01_data_understanding` | Carga, merge e inspeção inicial |
| 2 | `02_eda` | Distribuições, correlações, análise de fraude |
| 3 | `03_preprocessing` | Missings, encoding, normalização, balanceamento |
| 4 | `04_modeling` | Logistic Regression (baseline) vs XGBoost/LightGBM |
| 5 | `05_evaluation` | AUC-ROC, Precision-Recall, F1, SHAP values |

## Resultados

> *Em construção — será atualizado ao final do projeto.*

## Tecnologias

- **Python 3.11** com pandas, numpy, scikit-learn
- **XGBoost / LightGBM** — modelos de gradient boosting
- **SHAP** — interpretabilidade de modelos
- **imbalanced-learn** — técnicas para classes desbalanceadas
- **Jupyter Lab** — documentação interativa

## Aprendizados

> *Em construção — reflexões sobre decisões técnicas e trade-offs encontrados.*

---

*Desenvolvido por Fernando Daniel Marcelino como projeto de portfólio em Data Science.*
