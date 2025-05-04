import joblib
import pandas as pd
import streamlit as st
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler

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

    # Preencher nulos
    moda_cols = ['gender', 'enrolled_university', 'education_level', 'major_discipline', 'company_size', 'company_type']
    for col in moda_cols:
        moda = df[col].mode().iloc[0]
        df[col] = df[col].fillna(moda)

    median_cols = ['experience', 'last_new_job']
    for col in median_cols:
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)

    # GetDummies
    for col in ['gender', 'major_discipline', 'company_type']:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

    # TargetEncoder para city
    te = TargetEncoder(cols=['city'])
    te.fit(df, df['target'])  # usando o target para o fit
    df['city_target_encoded'] = te.transform(df)['city']

    # Tratar outliers de training_hours
    Q1 = df['training_hours'].quantile(0.25)
    Q3 = df['training_hours'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1 * IQR
    upper_bound = Q3 + 1 * IQR
    mask = (df['training_hours'] >= lower_bound) & (df['training_hours'] <= upper_bound)
    df = df[mask]  # importante: mantém consistência

    # Atualiza enrollee_id depois de remover outliers
    enrollee_id = df['enrollee_id']

    # Escalar numéricos
    numeric = ['experience', 'last_new_job', 'training_hours', 'city_target_encoded']
    scaler = StandardScaler()
    df[numeric] = scaler.fit_transform(df[numeric])

    features = df.drop(columns=['target', 'city', 'enrollee_id']).astype(float)
    labels = df['target']

    return features, labels, enrollee_id

# ---------------- APP ----------------

st.title("🎯 Predição de Probabilidade - Target = 1")

# Carregar modelo e dados
pipeline = joblib.load('final_model.pkl')
df_train = pd.read_csv('aug_train.csv')

# Mostrar dados brutos
st.subheader("📊 Dados brutos:")
st.dataframe(df_train.head())

# Processar
features, labels, enrollee_ids = treat_columns(df_train)
probs = pipeline.predict_proba(features)[:, 1]

# Juntar resultado
resultados = pd.DataFrame({
    'enrollee_id': enrollee_ids.values,
    'probabilidade_target_1': probs
})

# Ordenar
top_10 = resultados.sort_values(by='probabilidade_target_1', ascending=False).head(10)
bottom_10 = resultados.sort_values(by='probabilidade_target_1', ascending=True).head(10)

# Mostrar
st.subheader("🔝 Top 10 com mais probabilidade de ser 1:")
st.dataframe(top_10)

st.subheader("🔻 Top 10 com menos probabilidade de ser 1:")
st.dataframe(bottom_10)

# Botão de download
csv_result = resultados.to_csv(index=False).encode("utf-8")
st.download_button("📥 Baixar resultado CSV", csv_result, "resultados.csv", "text/csv")

