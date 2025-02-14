import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import plotly.express as px
import plotly.graph_objects as go

scaler = StandardScaler()

st.set_page_config(page_title='Modelo preditivo de ponto de virada', page_icon=':leftwards_arrow_with_hook:', layout='wide', initial_sidebar_state='collapsed')

# Cabeçalho
st.write("# Pós Tech - Data Analytics - 5DTAT")
st.write("## Datathon Fase 5")
st.write('### Modelo preditivo de Ponto de Virada')
st.write('''Resultados da análise preditiva realizada com base nos dados da ONG "Passos Mágicos". O objetivo principal foi prever a probabilidade de alunos atingirem o ponto de virada em 2025, utilizando dados históricos de 2020 a 2024 e modelos de machine learning.

O ponto de virada é um indicador importante de progresso educacional dos alunos, associado à sua evolução em múltiplos aspectos.
''')
st.write('Código disponível em: https://github.com/alexandreaquiles/postech-fiap-dtat-datathon-fase5')
st.divider()

spreadsheet_id = "1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0"

sheets = {
    "PEDE2022": "90992733",
    "PEDE2023": "555005642",
    "PEDE2024": "215885893"
}

def read_google_sheets(sheet_id):
  csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={sheet_id}"
  return pd.read_csv(csv_url, decimal=',')

def trata_dados_2022(df_2022):
    df_2022 = df_2022.rename(columns={"Idade 22": "Idade", "INDE 22": "INDE", "Defas": "Defasagem", "Matem": "Mat", "Portug": "Por", "Fase ideal": "Fase Ideal", "Pedra 22": "Pedra" })
    df_2022 = df_2022[['Fase', 'Idade', 'Gênero', 'Ano ingresso',
       'Pedra', 'INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'Mat', 'Por', 'IPV', 'IAN',
       'Defasagem']]
    df_2022["Ano"] = 2022
    df_2022["Gênero"] = df_2022["Gênero"].replace({
        "Menina": "Feminino",
        "Menino": "Masculino"
    })
    return df_2022

def trata_dados_2023(df_2023):
    df_2023 = df_2023.rename(columns={"INDE 2023": "INDE", "Pedra 2023": "Pedra"})
    df_2023 = df_2023[['Fase', 'Idade', 'Gênero', 'Ano ingresso',
       'Pedra', 'INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'Mat', 'Por', 'IPV', 'IAN',
       'Defasagem']]
    df_2023["Ano"] = 2023
    df_2023 = df_2023.dropna()
    df_2023 = df_2023[df_2023['INDE'] != "#DIV/0!"].reset_index(drop=True)

    df_2023["Idade"] = df_2023["Idade"].str.replace(r"^1/", "", regex=True)  
    df_2023["Idade"] = df_2023["Idade"].str.replace(r"/1900$", "", regex=True)  
    df_2023["Idade"] = pd.to_numeric(df_2023["Idade"], errors="coerce")

    df_2023['INDE'] = pd.to_numeric(df_2023['INDE'].str.replace(",", "."))
    df_2023['IPS'] = pd.to_numeric(df_2023['IPS'].str.replace(",", "."))
    df_2023['IDA'] = pd.to_numeric(df_2023['IDA'].str.replace(",", "."))

    df_2023['Fase'] = df_2023['Fase'].apply(lambda val: 0 if val == 'ALFA' else int(val.split(' ')[1]))

    return df_2023

def trata_dados_2024(df_2024):
    df_2024 = df_2024.rename(columns={"INDE 2024": "INDE", "Pedra 2024": "Pedra"})
    df_2024 = df_2024[['Fase', 'Idade', 'Gênero', 'Ano ingresso',
       'Pedra', 'INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'Mat', 'Por', 'IPV', 'IAN',
       'Defasagem']]
    df_2024["Ano"] = 2024
    df_2024 = df_2024.dropna()
    df_2024['INDE'] = pd.to_numeric(df_2024['INDE'].str.replace(",", "."))
    df_2024['IDA'] = pd.to_numeric(df_2024['IDA'].str.replace(",", "."))

    df_2024['Fase'] = df_2024['Fase'].apply(lambda val: 0 if val == 'ALFA' else int(''.join(filter(str.isdigit, val))))
    
    return df_2024

@st.cache_data
def carrega_dados():
    df_2022 = read_google_sheets(sheets["PEDE2022"])
    df_2023 = read_google_sheets(sheets["PEDE2023"])
    df_2024 = read_google_sheets(sheets["PEDE2024"])

    df_2022 = trata_dados_2022(df_2022)

    df_2023 = trata_dados_2023(df_2023)

    df_2024 = trata_dados_2024(df_2024)

    df_final = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)

    return df_final

def codifica_variaveis_categoricas(df_final):

    #  categorias sem relação de ordem com poucos valores
    df_final = pd.get_dummies(df_final, columns=['Gênero'])

    #  categorias sem relação de ordem

    le = LabelEncoder()
    df_final['Pedra'] = le.fit_transform(df_final['Pedra'])

    return df_final

def calcula_atingiu_pv(df_final):
    media_ipv = df_final['IPV'].mean()
    desvio_padrao_ipv = df_final['IPV'].std()
    IPV_M_D = media_ipv + desvio_padrao_ipv
    df_final['Atingiu_PV'] = np.where(df_final['IPV'] > IPV_M_D, 1, 0)
    return df_final

def treina_modelo(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Training the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

df_final = carrega_dados()

df_final = codifica_variaveis_categoricas(df_final)

df_final = calcula_atingiu_pv(df_final)

X = df_final.drop(columns=["Atingiu_PV", "IPV"])  # Removendo variável target e variável correlacionada
y = df_final["Atingiu_PV"]

model, X_test, y_test = treina_modelo(X, y)

tab1, tab2 = st.tabs(["Avaliação do Modelo", "Predição"])

with tab1:

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    st.write("### Acurácia")
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Acurácia do modelo", value=f"{accuracy:.4f}")

    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("### Relatório de Classificação")
    st.dataframe(pd.DataFrame(report).transpose())

    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write("### Matriz de Confusão")
    fig_conf_matrix = px.imshow(conf_matrix, text_auto=True, x=["Não Atingiu", "Atingiu"], y=["Não Atingiu", "Atingiu"], color_continuous_scale="Blues")
    st.plotly_chart(fig_conf_matrix)

    # Feature importance
    importances = model.feature_importances_
    df_feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    st.write("### Importância das Features")
    fig_importance = px.bar(df_feature_importance, x="Importance", y="Feature", orientation='h', title="Importância das Features", color="Importance")
    st.plotly_chart(fig_importance)

    with tab2:
        st.write("### Preencha os valores das features para prever `Atingiu_PV`")
        
        user_input = {}
        for feature in X.columns:
            user_input[feature] = st.number_input(f"Digite o valor para {feature}", value=0.0)

        if st.button("Prever"):
            input_data = pd.DataFrame([user_input])

            input_data = scaler.transform(input_data)

            prediction = model.predict(input_data)
            result = "Atingiu" if prediction[0] == 1 else "Não Atingiu"

            # Display the prediction result
            st.info(f"Previsão: {result}")