import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
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

st.set_page_config(layout="wide")
st.title("Classificador Binário")

menu = st.sidebar.selectbox("Escolha uma opção", [
    "Entenda os dados",
    "Preveja se um candidato está em busca de emprego",
    "Busque um candidato por seu ID",
    "Veja quais são os candidatos mais prováveis a mudar de emprego",
    "Entenda a escolha do modelo"
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
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div style='text-align: justify'><h5>Os dados utilizados são referentes à intenção de profissionais da área de dados em trocar de emprego. Para que seja determinada a probabilidade do profissional estar interessado em mudar de emprego, foram analisados dados de gênero, formação e experiência profissional, assim como dados das vagas que ocupavam no momento da coleta dos dados.</h5></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: justify'><h5>A partir disso, foi possível determinar um modelo de classificação binária, que mostrou desempenho de XXX em acurácia para o conjunto geral dos dados. Também não foi identificado overfitting no modelo escolhido. Para saber mais sobre a escolha do modelo e seu desempenho, acesse a página XXX</h5></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<h5>Cerca de <span style='color:#E07A5F;'>25%</span> <br> dos candidatos <span style='color:#E07A5F;'><b>estão <br> em busca de um novo emprego</b></span></h5>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h5>As intenções dos candidatos <br><span style='color:#E07A5F;'><b>não</b></span> são afetadas pelo seu <span style='color:#E07A5F;'>gênero</span>, <br><span style='color:#E07A5F;'>área de formação</span> e <span style='color:#E07A5F;'>horas de treinamento</span></h5>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col3, col4 = st.columns([1.5, 1])

    with col3:
        st.markdown("<h5>Candidatos que <span style='color:#E07A5F;'>saíram do <br> seu último emprego há mais tempo</span>, <br> têm <span style='color:#E07A5F;'><b>menor chance</b></span> de trocar de emprego</h5>", unsafe_allow_html=True)

        # Dados
        df1 = pd.DataFrame({
            'x' : ['Nunca trabalhou', '1', '2', '3', '4', 'Mais de 4'],
            'y' : [30.1387, 26.4303, 24.1379, 22.5586, 22.1574, 18.2371]
            })

        fig, ax = fig, ax = plt.subplots(figsize=(6, 3.75), dpi=300)
        ax.plot(df1['x'], df1['y'], linestyle='-', marker='o', color='#f15050ff')
        ax.set_xlabel('Anos desde o último emprego')
        ax.set_ylabel('Porcentagem de candidatos \n em busca de empregos')
        ax.set_ylim(0, 35)
        ax.set_yticks([])
        ax.grid(False)
        fig.tight_layout()

        # Adicionar os valores acima de cada ponto
        for i in range(len(df1)):
            ax.annotate(f"{df1['y'][i]:.0f}",
                    (df1['x'][i], df1['y'][i]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha='center', color='#444')

        st.pyplot(fig)

        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        st.markdown("<h5>Os candidatos <span style='color:#E07A5F;'>graduados</span> buscam <br> por novos empregos com <span style='color:#E07A5F;'><b>mais frequência</b></span></h5>", unsafe_allow_html=True)

        # Bloco 2: Escolaridade
        x = ['Ensino Fundamental', 'Ensino Médio', 'Graduação', 'Mestrado', 'Doutorado']
        y = [13, 20, 28, 21, 14]
        cores = ['#e0dede', '#f9a3a3', '#f15050ff', '#f77c7c', '#e0dede']
        y_pos = np.arange(len(x))
    
        # Criar figura e eixo
        fig, ax = plt.subplots(figsize=(6, 3.75), dpi=300)
        bars = ax.bar(y_pos, y, color=cores)
    
        # Adicionar nomes no eixo X
        ax.set_xticks(y_pos)
        ax.set_xticklabels(x, rotation=15, ha='right')
        ax.set_ylim(0, 30)
        ax.set_yticks([])
        ax.set_ylabel('Porcentagem de candidatos\n em busca de empregos')
        ax.grid(False)
    
        # Mostrar valores acima das barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', color='#444')
    
        fig.tight_layout()
    
        # Mostrar no Streamlit
        st.pyplot(fig)

        st.markdown("<br><br><br>", unsafe_allow_html=True)

        st.markdown("<h5>Quanto <span style='color:#E07A5F;'>maior o IDH</span> da <br> cidade em que se localiza a empresa, <br> <span style='color:#E07A5F;'><b>menos candidatos</b></span> buscam por novos empregos</h5>", unsafe_allow_html=True)
        
        # Bloco 3: IDH
        x = ['Baixo', 'Médio', 'Alto']
        y = [58.7, 48.7, 16.5]
    
        # Plot
        fig, ax = plt.subplots(figsize=(6, 3.75), dpi=300)
        ax.plot(x, y, marker='o', linewidth=2, color='#f15050ff')
        ax.fill_between(x, y, color='#f9a3a3', alpha=0.3)
        ax.set_ylim(0, 70)
        ax.set_yticks([])
        ax.set_xlabel('IDH da cidade')
        ax.set_ylabel('Porcentagem de candidatos\n em busca de empregos')
        ax.grid(False)
        fig.tight_layout()
    
        for i, val in enumerate(y):
            ax.text(x[i], val, f'{val:.0f}', ha='center', va='bottom')
    
        st.pyplot(fig)



    with col4:
        st.markdown("<br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        
        st.markdown("<h5>E candidatos <span style='color:#E07A5F;'>sem <br> experiência prévia</span> relevante, <br> procuram <span style='color:#E07A5F;'><b>mais</b></span> por um emprego</h5>", unsafe_allow_html=True)
        
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        
        st.markdown("<h5>E candidatos que <span style='color:#E07A5F;'>estudaram <br> em tempo integral</span>, também</h5>", unsafe_allow_html=True)

        st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        
        st.markdown("<h5>Já o <span style='color:#E07A5F;'>tipo e tamanho</span><br> da empresa, em <span style='color:#E07A5F;'><b>pouco <br> influencia</b></span> a busca de empregos</h5>", unsafe_allow_html=True)

elif menu == "Preveja se um candidato está em busca de emprego":
    st.subheader("Preveja se um candidato está em busca de emprego")

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
    st.subheader("Busque um candidato por seu ID")
    id_input = st.text_input("Digite o ID (de 1 a 33379)")

    if st.button("Buscar"):
        filt = resultados[resultados['ID candidato'] == int(id_input)]
        if not filt.empty:
            prob2 = filt.iloc[0]['Probabilidade']
            st.write(f"A probabilidade do candidato de ID {id_input} estar em busca de um novo emprego é: {prob2*100:.0f}%")
        else:
            st.error("ID não encontrado nos resultados.")

elif menu == "Veja quais são os candidatos mais e menos prováveis a mudar de emprego":
    st.subheader("Veja quais são os candidatos mais e menos prováveis a mudar de emprego")

    # Criação dos DataFrames
    top_10 = resultados.nlargest(10, 'Probabilidade')[['ID candidato', 'Probabilidade']]
    bottom_10 = resultados.nsmallest(10, 'Probabilidade')[['ID candidato', 'Probabilidade']]

    # Formatação das probabilidades como porcentagem
    top_10['Probabilidade (%)'] = (top_10['Probabilidade'] * 100).round(0).astype(int).astype(str) + '%'
    bottom_10['Probabilidade (%)'] = (bottom_10['Probabilidade'] * 100).round(0).astype(int).astype(str) + '%'

    st.write("Os candidatos mais prováveis")
    st.dataframe(top_10[['ID candidato', 'Probabilidade (%)']].reset_index(drop=True))

    st.write("Os candidatos menos prováveis")
    st.dataframe(bottom_10[['ID candidato', 'Probabilidade (%)']].reset_index(drop=True))

elif menu == "Entenda a escolha do modelo":
