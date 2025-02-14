# postech-fiap-dtat-datathon-fase5

Entrega do Datathon da Fase 5 da Pós Tech em Data Analytics da FIAP, turma 5DTAT.

## Análise Preditiva do Ponto de Virada da ONG Passos Mágicos

Desenvolvemos um Dashboard Interativo que torna possível verificar os resultados de avaliação e interagir com um modelo de Machine Learning preditivo, com o intuito de fazer a previsão do atingimento ou não do Ponto de Virada.

O ponto de virada é um indicador importante de progresso educacional dos alunos, associado à sua evolução em múltiplos aspectos.

O Dashboard pode ser encontrado em: https://postech-fiap-dtat-datathon-fase5.streamlit.app

### Quem somos nós?

Somos da Consultoria **Grupo 46**, uma consultoria especializada em criação de modelos preditivos.

Integrantes:

- Alexandre Aquiles Sipriano da Silva (alexandre.aquiles@gmail.com)
- Gabriel Machado Costa (gabrielmachado2211@gmail.com)
- Caio Martins Borges (caio.borges@bb.com.br)

## Relatório Técnico

Este repositório contém a análise desenvolvida para a Fase 5 do Datathon, utilizando dados históricos diferentes turmas da ONG Educacional Passos Mágicos e técnicas de aprendizado de máquina para previsões.

### 📦 Importação de Bibliotecas

Pandas, Seaborn, NumPy, Matplotlib.

Modelos de machine learning com o Scikit-Learn, como o RandomForestClassifier.

### 📥 Carregamento de Dados

Dados obtidos a partir de planilhas Google Sheets, referentes aos anos de 2022, 2023 e 2024.

### 🔍 Análise Exploratória

Análise inicial com head(), shape(), info() e describe().

Verificação de valores nulos e distribuição de variáveis.

Tratamento de valores ausentes com imputação adequada.

Codificação de variáveis categóricas com One Hot Enconder e LabelEncoder.

Normalização de variáveis numéricas utilizando StandardScaler.

### 🧠 Modelagem Preditiva

Utilização do modelo RandomForestClassifier.

Pré-processamento dos dados com LabelEncoder e StandardScaler.

Avaliação de métricas de desempenho, como precisão, recall, matriz de confusão e feature importance.

### ⚙️ Deploy do modelo

O modelo foi disponibilizado através de uma aplicação interativa utilizando o Streamlit.

O Streamlit permite a visualização e interação com os resultados das previsões de maneira simples e acessível utilizando Plotly Express.

O deploy foi realizado em uma plataforma como Streamlit Cloud, facilitando o acesso por meio do seguinte link público: https://postech-fiap-dtat-datathon-fase5.streamlit.app


### 🛠️ Melhorias Futuras

Explorar outras técnicas de modelagem, como redes neurais.

Refinar o pré-processamento dos dados, considerando variáveis categóricas específicas.

### Como rodar localmente?

Para rodar localmente, é necessário ter o Python 3+ e instalar todas as dependências com o comando:

```sh
pip install -r requirements.txt
```

Em seguida, precisamos executar a CLI do Streamlit:

```sh
streamlit run Dashboard.py
```

Para 

Para executar o notebook localmente, recomenda-se executar no Jupyter o seguinte comando:

```sh
jupyter notebook 5dtat_datathon_fase5.ipynb
```
