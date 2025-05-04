import joblib
import pandas as pd
import streamlit as st
from collections import Counter
from collections import defaultdict
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def treat_columns(df_train):
    df = df_train.copy()

    # Salvar o ID (enrollee_id) antes de qualquer transformação
    enrollee_id = df['enrollee_id']

    # city: retirar o "city" do ID da cidade
    df['city'] = df['city'].str.extract(r'city_(\d+)').astype(int)

    # relevent_experience: transformar em binário
    df['relevent_experience'] = df['relevent_experience'].replace({
        'Has relevent experience': 1,
        'No relevent experience': 0
        }).astype('Int64')

    # enrolled_university: transformar em ordinal
    df['enrolled_university'] = df['enrolled_university'].replace({
        'no_enrollment': 0,
        'Part time course': 1,
        'Full time course': 2
        }).astype('Int64')

    # education_level: transformar em ordinal
    df['education_level'] = df['education_level'].replace({
        'Primary School' : 0,
        'High School' : 1,
        'Graduate' : 2,
        'Masters' : 3,
        'Phd' : 4
        }).astype('Int64')

    # experience: transformar em numérico
    df['experience'] = df['experience'].str.extract(r'(\d+)').astype('Int64')

    # company_size: transformar em numérico
    df['company_size'] = df['company_size'].replace({
        '<10': 0,
        '10/49': 1,
        '50-99': 2,
        '100-500': 3,
        '500-999' : 4,
        '1000-4999': 5,
        '5000-9999': 6,
        '10000+': 7
        }).astype('Int64')

    # last_new_job: transformar em numérico
    df['last_new_job'] = pd.to_numeric(df['last_new_job'].replace({'never': 0, '>4': 5}), errors='coerce').astype('Int64')

    # Colunas com moda
    moda_cols = ['gender', 'enrolled_university', 'education_level', 'major_discipline', 'company_size', 'company_type']
    for col in moda_cols:
        moda = df[col].mode().iloc[0]
        df[col] = df[col].fillna(moda)

    # Coluna com mediana
    median_cols = ['experience', 'last_new_job']
    for col in median_cols:
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)

    # GetDummies para nominais
    for col in ['gender', 'major_discipline', 'company_type']:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

    # TargetEncoder para city
    te = TargetEncoder(cols=['city'])
    te.fit(df, df['target'])

    # transforma o df
    df['city_target_encoded'] = te.transform(df)['city']

    # tratar outliers de training_hours
    Q1 = df['training_hours'].quantile(0.25)  # 1º quartil
    Q3 = df['training_hours'].quantile(0.75)  # 3º quartil
    IQR = Q3 - Q1  # Amplitude interquartílica (IQR)
    lower_bound = Q1 - 1 * IQR  # Limite inferior
    upper_bound = Q3 + 1 * IQR  # Limite superior

    df = df[(df['training_hours'] >= lower_bound) & (df['training_hours'] <= upper_bound)]

    numeric = ['experience', 'last_new_job', 'training_hours', 'city_target_encoded']
    scaler = StandardScaler()
    df[numeric] = scaler.fit_transform(df[numeric])

    features = df.drop(columns=['target', 'city', 'enrollee_id']).astype(float)
    labels = df['target']

    # Se o ID foi necessário, retornar junto com as features
    return features, labels, enrollee_id


# Carregar modelo
pipeline = joblib.load('final_model.pkl')

# Carregar df
df_train = pd.read_csv('aug_train.csv')

# Fazer predição (obter probabilidades de 1 no target)
#pred_probs = pipeline.predict_proba(features)[:, 1]  # Probabilidade da classe 1

# Associar as probabilidades com o ID
#pred_prob_df = pd.DataFrame({
#    'enrollee_id': enrollee_id,
#    'probabilidade_1': pred_probs
#})

# Ordenar para pegar as 10 maiores e menores probabilidades de ser 1
#top_10 = pred_prob_df.sort_values(by='probabilidade_1', ascending=False).head(10)
#bottom_10 = pred_prob_df.sort_values(by='probabilidade_1', ascending=True).head(10)

#print("Top 10 com mais probabilidade de ser 1:")
#print(top_10)

#print("\nTop 10 com menos probabilidade de ser 1:")
#print(bottom_10)

st.title("Predição de Probabilidade - Target = 1")

st.write("Dados brutos:")
st.dataframe(df_train.head())

# Salvar o enrollee_id antes do tratamento
enrollee_ids = df_train['enrollee_id'].copy()

# Tratar colunas
features, labels, enrollee_id = treat_columns(df_train)

# Gerar probabilidades
probs = pipeline.predict_proba(features)[:, 1]

# Juntar com o ID
resultados = pd.DataFrame({
    'enrollee_id': enrollee_ids.values,
    'probabilidade_target_1': probs
})

# Mostrar
st.write("Resultados com probabilidades:")
st.dataframe(resultados)

# Download
csv_result = resultados.to_csv(index=False).encode("utf-8")
st.download_button("📥 Baixar resultado CSV", csv_result, "resultados.csv", "text/csv")
