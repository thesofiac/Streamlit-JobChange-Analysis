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
    try:
        df['city_target_encoded'] = te.transform(df)['city']

    except NameError:
        te = TargetEncoder(cols=['city'])
        te.fit(df, df['target'])  # usando o target para o fit
        df['city_target_encoded'] = te.transform(df)['city']

    # Tratar outliers de training_hours
    try:
        mask = (df['training_hours'] >= lower_bound) & (df['training_hours'] <= upper_bound)
        df = df[mask]  # importante: mantém consistência

    except:
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
    try:
        df[numeric] = scaler.transform(df[numeric])

    except:
        scaler = StandardScaler()
        scaler.fit(df[numeric])
        df[numeric] = scaler.transform(df[numeric])

    features = df.drop(columns=['target', 'city', 'enrollee_id']).astype(float)
    labels = df['target']

    return features, labels, enrollee_id

# Carregar modelo e dados
pipeline = joblib.load('final_model.pkl')
df_train = pd.read_csv('aug_train.csv')

# Processar
features, labels, enrollee_ids = treat_columns(df_train)
probs = pipeline.predict_proba(features)[:, 1]
features_cols = features.columns

# Juntar resultado
resultados = pd.DataFrame({
    'ID candidato': enrollee_ids.values,
    'Probabilidade': probs
})

st.title("Classificador Binário")

menu = st.sidebar.selectbox("Escolha uma opção", [
    "Entenda os dados",
    "Preveja se um candidato está em busca de emprego",
    "Busque um candidato por seu ID",
    "Veja quais são os candidatos mais prováveis a mudar de emprego"
])

dic_gender = {
        'Feminino' : 'Female',
        'Masculino' : 'Male',
        'Outros' : 'Other'
        }

dic_scholarity = {
        'Ensino Fundamental' : 'Primary School',
        'Ensino Médio' : 'High School',
        'Graduação' : 'Graduate',
        'Mestrado' : 'Masters',
        'Doutorado' : 'Phd'
        }

dic_major = {
        'Negócios' : 'Business Degree',
        'Artes' : 'Arts',
        'Humanas' : 'Humanities',
        'Ciência e Tecnologia' : 'STEM',
        'Sem Formação' : 'No Major',
        'Outros' : 'Other'
        }

dic_rel_exp = {
        'Sim' : 'Has relevent experience',
        'Não' : 'No relevent experience'
        }

dic_last_job = {
        'Nunca trabalhou' : 'never',
        '1 ano' : '1',
        '2 anos' : '2',
        '3 anos' : '3',
        '4 anos' : '4',
        'Acima de 4 anos' : '>4'
        }

dic_course = {
        'Curso em Tempo Integral' : 'Full time course',
        'Curso em Meio Período' : 'Part time course',
        'Não realizou curso' : 'no_enrollment'
        }

dic_company_type = {
        'Empresa Privada' : 'Pvt Ltd',
        'Empresa Pública' : 'Public Sector',
        'Startup' : 'Funded Startup',
        'Startup em Estágio Inicial' : 'Early Stage Startup',
        'ONG' : 'NGO',
        'Outros' : 'Other'
        }

dic_company_size = {
        'Até 10 funcionários' : '<10',
        'De 10 a 49 funcionários' : '10/49',
        'De 50 a 99 funcionários' : '50-99',
        'De 100 a 500 funcionários' : '100-500',
        'De 500 a 999 funcionários' : '500-999',
        'De 1000 a 4999 funcionários' : '1000-4999',
        'De 5000 a 9999 funcionários' : '5000-9999',
        'Mais de 10000 funcionários' : '10000+'
        }

gender_options = list(dic_gender.keys())
scholarity_options = list(dic_scholarity.keys())
major_options = list(dic_major.keys())
relevant_experience = list(dic_rel_exp.keys())
lastjob_options = list(dic_last_job.keys())
course_options = list(dic_course.keys())
company_options = list(dic_company_type.keys())
company_size_options = list(dic_company_size.keys())


if menu == "Entenda os dados":
    st.subheader("Entenda os dados")

elif menu == "Preveja se um candidato está em busca de emprego":
    st.subheader("Adicione as informações do candidato e da empresa")

    f1 = st.selectbox("Gênero", gender_options)
    f2 = st.selectbox("Escolaridade", scholarity_options)
    f3 = st.selectbox("Área de formação", major_options)
    f4 = st.number_input("Anos de experiência", value=0.0)
    if int(f4) > 20:
        f4 = '>20'
    elif int(f4) < 1:
        f4 = '<1'
    else:
        f4 = str(f4)

    f5 = st.selectbox("Já teve experiência na área?", relevant_experience)
    f6 = st.selectbox("Quantos anos desde o último emprego?", lastjob_options)
    f7 = st.selectbox("Qual o período do curso realizado?", course_options)
    f8 = st.number_input("Quantas horas de treinamento realizadas?", value=0.0)

    f9 = st.number_input("Código da cidade em que se localiza a empresa", value=0.0)
    f9 = 'city_' + str(f9)

    f10 = st.number_input("IDH da cidade", value=0.0)
    f11 = st.selectbox("Tipo da empresa", company_options)
    f12 = st.selectbox("Tamanho da empresa", company_size_options)

    f13 = '0'
    f14 = '0'

    if st.button("Prever"):
        input_df = pd.DataFrame([[int(f13), str(f9), float(f10), dic_gender[f1], dic_rel_exp[f5], dic_course[f7], dic_scholarity[f2], dic_major[f3], str(f4), dic_company_size[f12], dic_company_type[f11], dic_last_job[f6], int(f8), int(f14)]], columns=['enrollee_id', 'city', 'city_development_index', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours', 'target'])
        features1, labels1, enrollee_ids1 = treat_columns(input_df)
        features2 = features1.reindex(columns=features_cols, fill_value=0)
        prob1 = pipeline.predict_proba(features2)[0, 1]
        prediction = pipeline.predict(features2)[0]
        st.write(f"A probabilidade do candidato estar em busca de um novo emprego é: {prob1*100:.0f}%")

elif menu == "Busque um candidato por seu ID":
    st.subheader("Buscar dados por ID")
    id_input = st.text_input("Digite o ID (de 1 a 33379)")

    if st.button("Buscar"):
        filt = resultados[resultados['ID candidato'] == int(id_input)]
        if not filt.empty:
            prob2 = filt.iloc[0]['Probabilidade']
            st.write(f"A probabilidade do candidato de ID {id_input} estar em busca de um novo emprego é: {prob2*100:.0f}%")
        else:
            st.error("ID não encontrado nos resultados.")

elif menu == "Veja quais são os candidatos mais prováveis a mudar de emprego":
    st.subheader("Os candidatos mais e menos prováveis de mudar de emprego")
    top_10 = resultados.nlargest(10, 'Probabilidade')[['ID candidato', 'Probabilidade']]
    bottom_10 = resultados.nsmallest(10, 'Probabilidade')[['ID candidato', 'Probabilidade']]

    st.write("Os candidatos mais prováveis")
    st.dataframe(top_10.reset_index(drop=True))

    st.write("Os candidatos menos prováveis")
    st.dataframe(bottom_10.reset_index(drop=True))
