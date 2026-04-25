# 🏢 End-to-End Insurance Commission Prediction Pipeline

Bem-vindo(a) ao repositório do **Modelo de Previsão de Comissão**. Este projeto apresenta um pipeline de Data Science completo (End-to-End) desenhado especificamente para o mercado de seguros, visando otimizar precificações e prever a elasticidade das comissões praticadas por corretores no momento das **renovações de apólices**.

O grande diferencial deste projeto reside na adoção de uma arquitetura híbrida de Machine Learning (Ensemble Guiado por Regras de Negócio), cruzando uma abordagem de _Classificação_ com uma de _Regressão_ para maximizar as garantias financeiras da corretora e amortecer erros da inteligência artificial. Todo o ecossistema é validado rigorosamente por uma poderosa Suite de Geração de Relatórios Corporativos em **PDF, HTML e Markdown**.

---

## 🎯 Objetivo do Negócio

Dada a apólice anterior de um cliente (informando Prêmio, Diferenças de Preço, Históricos de RF/Serasa/Sinistro, percentual de comissão original, etc.), o framework responde:
1. **Predidor Direcional:** Ele Aumentará, Diminuirá ou Manterá a sua margem de comissão?
2. **Predidor Vetorial:** Qual será o exato valor percentual (`%`) dessa nova comissão aplicada?
3. **Fusão Automática Combinada:** Qual é o valor validado após cruzar as decisões (salvando discrepâncias) e aplicar heurísticas vitais do estatístico de negócio?

---

## ⚙️ Arquitetura e Estrutura do Projeto

O projeto está organizado em uma estrutura modular para facilitar a manutenção e escalabilidade:

### 📁 Pastas Principais
- **`scripts/`**: Contém todo o código-fonte Python do pipeline (01 a 07).
- **`data/`**: Armazena as bases de dados brutas (`.csv`) e bases processadas/preditas (`.parquet`).
- **`reports/`**: Centraliza todos os relatórios gerados (PDF, HTML, MD) e a pasta `images/` com todos os gráficos analíticos.

---

## 🚀 Execução do Pipeline

A jornada dos dados engloba as seguintes etapas rigidamente separadas por scripts sequenciais modulares localizados em `scripts/`:

1.  **`01_data_generation.py`**: Gera simulativamente **200.000 perfis paramétricos** com lógica de mercado.
2.  **`02_eda_and_preprocessing.py`**: Realiza limpeza, imputação e Feature Engineering (PCA).
3.  **`03_classification_modeling.py`**: Treina o classificador direcional (Random Forest) e gera o primeiro laudo técnico.
4.  **`04_regression_modeling.py`**: Treina o regressor (Ridge Regression) para estimativa monetária fina.
5.  **`05_ensemble_evaluation.py`**: Aplica a lógica de Ensemble (regras de negócio) e consolida a base final.
6.  **`06_export_class_images.py`**: Exporta gráficos de classificação em alta definição para a pasta `reports/images/`.
7.  **`07_generate_final_pdf.py`**: Consolida o **Relatório Executivo Final** (`reports/RELATORIO_EXECUTIVO_FINAL.pdf`) integrando todas as visões do projeto.

---

## 🛠️ Como Executar Localmente

1. Tenha o **Python 3.10+** ativo.
2. Ative um *Virtual Environment* e instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o pipeline completo:
   ```bash
   python scripts/01_data_generation.py
   python scripts/02_eda_and_preprocessing.py
   python scripts/03_classification_modeling.py
   python scripts/04_regression_modeling.py
   python scripts/05_ensemble_evaluation.py
   python scripts/06_export_class_images.py
   python scripts/07_generate_final_pdf.py
   ```

---

## 📊 Entregáveis de Negócio

Localizados na pasta **`reports/`**:
- **`RELATORIO_EXECUTIVO_FINAL.pdf`**: Laudo consolidado para diretoria.
- **`Relatorio_Classificacao.pdf`**: Detalhamento técnico da IA Direcional.
- **`Relatorio_Regressao.pdf`**: Detalhamento técnico da IA Vetorial.
- **`Relatorio_Combinado_Ensemble.pdf`**: Defesa da lógica de regras e redução de erro.

Base Final de Consumo em **`data/BASE_OUTPUT_FINAL_PRONTA.parquet`**.

---
