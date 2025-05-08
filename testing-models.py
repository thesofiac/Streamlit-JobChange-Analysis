import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
…from collections import Counter
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

df_train = pd.read_csv('aug_train.csv')

# Técnicas de balanceamento
tecnicas = {
      'Sem Balanceamento': None,
      'SMOTE': SMOTE(random_state=42),
      'ADASYN': ADASYN(random_state=42),
      'Random Over-Sampling': RandomOverSampler(random_state=42),
      'Random Under-Sampling': RandomUnderSampler(random_state=42)
    }

# Modelos disponíveis
available_models = {
      "LogisticRegression": LogisticRegression(solver='saga', max_iter=1000, random_state=42),
      "RandomForestClassifier": RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42),
      "XGBClassifier": XGBClassifier(random_state=42)
    }

models = ['LogisticRegression',  'RandomForestClassifier', 'XGBClassifier']

# Parâmetros para LogisticRegression
param_grid_lr = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['saga'],
    'clf__max_iter': [1000],
    'clf__random_state': [42]
}

# Parâmetros para RandomForestClassifier
param_grid_rf = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5, 10],
    'clf__random_state': [42]
}

# Parâmetros para XGBClassifier
param_grid_xgb = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 6, 10],
    'clf__learning_rate': [0.01, 0.1, 0.2],
    'clf__subsample': [0.8, 1.0],
    'clf__random_state': [42]
}

df = df_train.copy()

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

X = df.drop(columns=['target'])
y = df['target']

# Dicionário para armazenar os resultados
resultados_finais = defaultdict(list)

random_states = [0, 42, 100, 123, 2024]
#random_states = [42]
for random_state in random_states:
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

      df_train = pd.concat([X_train, y_train], axis=1)
      df_test = pd.concat([X_test, y_test], axis=1)

      df = df_train.copy()
      df1 = df_test.copy()
      fill_values = {}

      # Colunas com moda
      moda_cols = ['gender', 'enrolled_university', 'education_level', 'major_discipline', 'company_size', 'company_type']
      for col in moda_cols:
        moda = df[col].mode().iloc[0]
        df[col] = df[col].fillna(moda)
        fill_values[col] = moda

      # Coluna com mediana
      median_cols = ['experience', 'last_new_job']
      for col in median_cols:
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)
        fill_values[col] = mediana

      # Tratar dados do teste
      for col, value in fill_values.items():
        df1[col] = df1[col].fillna(value)

      # GetDummies para nominais
      for col in ['gender', 'major_discipline', 'company_type']:
        df = pd.get_dummies(df, columns=[col], drop_first=True)
        df1 = pd.get_dummies(df1, columns=[col], drop_first=True)

      te = TargetEncoder(cols=['city'])
      te.fit(df, df['target'])

      # transforma o df
      df['city_target_encoded'] = te.transform(df)['city']
      df1['city_target_encoded'] = te.transform(df1)['city']

      df = df.copy()
      df1 = df1.copy()

      # tratar outliers de training_hours
      Q1 = df['training_hours'].quantile(0.25)  # 1º quartil
      Q3 = df['training_hours'].quantile(0.75)  # 3º quartil
      IQR = Q3 - Q1  # Amplitude interquartílica (IQR)
      lower_bound = Q1 - 1 * IQR  # Limite inferior
      upper_bound = Q3 + 1 * IQR  # Limite superior

      df = df[(df['training_hours'] >= lower_bound) & (df['training_hours'] <= upper_bound)]
      df1 = df1[(df1['training_hours'] >= lower_bound) & (df1['training_hours'] <= upper_bound)]

      numeric = ['experience', 'last_new_job', 'training_hours', 'city_target_encoded']
      scaler = StandardScaler()
      df[numeric] = scaler.fit_transform(df[numeric])
      df1[numeric] = scaler.transform(df1[numeric])

      train_features = df.drop(columns=['target','city','enrollee_id']).astype(float)
      train_labels = df['target']

      test_features = df1.drop(columns=['target','city','enrollee_id']).astype(float)
      test_labels = df1['target']

      # Verificar o equilíbrio das classes
      class_count = Counter(train_labels)
      majority_class = max(class_count, key=class_count.get)
      minority_class = min(class_count, key=class_count.get)
      majority_class_count = class_count[majority_class]
      minority_class_count = class_count[minority_class]

      # Verificar se o ADASYN é apropriado (não vai funcionar se o número de vizinhos for muito pequeno)
      if majority_class_count / minority_class_count < 10:
        # Se o desequilíbrio for muito grande, desabilitar o ADASYN
        tecnicas['ADASYN'] = None

      for m in models:
          model = available_models.get(m)

          for nome, tecnica in tecnicas.items():
            if tecnica is None:
              pipeline = Pipeline([
                  ('remove_constantes', VarianceThreshold(threshold=0.0)),
                  ('selecao', SelectKBest(score_func=f_classif, k=10)),
                  ('clf', model)
                  ])
            else:
              pipeline = Pipeline([
                  ('remove_constantes', VarianceThreshold(threshold=0.0)),
                  ('selecao', SelectKBest(score_func=f_classif, k=10)),
                  ('balanceamento', tecnica),
                  ('clf', model)
                  ])

            # Testar GridSearch
            scorer = make_scorer(recall_score, pos_label=1)

            if m == 'LogisticRegression':
              grid = GridSearchCV(
                  estimator=pipeline,
                  param_grid=param_grid_lr,
                  scoring=scorer,
                  cv=5,
                  n_jobs=-1,
                  verbose=1
                  )

            elif m == 'RandomForestClassifier':
              grid = GridSearchCV(
                  estimator=pipeline,
                  param_grid=param_grid_rf,
                  scoring=scorer,
                  cv=5,
                  n_jobs=-1,
                  verbose=1
                  )

            elif m == 'XGBClassifier':
              grid = GridSearchCV(
                  estimator=pipeline,
                  param_grid=param_grid_xgb,
                  scoring=scorer,
                  cv=5,
                  n_jobs=-1,
                  verbose=1
                  )

            grid.fit(train_features, train_labels)

            best_model = grid.best_estimator_
            probas = best_model.predict_proba(test_features)[:, 1]
            threshold = 0.4
            labels_pred = (probas >= threshold).astype(int)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # 4. Calcula métricas do teste
            acuracia_test = accuracy_score(test_labels, labels_pred)
            precisao_test = precision_score(test_labels, labels_pred, average='macro')
            recall_test = recall_score(test_labels, labels_pred, average='macro')
            f1_test = f1_score(test_labels, labels_pred, average='macro')

            # 5. Para avaliar no treino com validação cruzada, use diretamente o best_model:
            acuracia = np.mean(cross_val_score(best_model, train_features, train_labels, cv=skf, scoring='accuracy'))
            precisao = np.mean(cross_val_score(best_model, train_features, train_labels, cv=skf, scoring='precision_macro'))
            recall = np.mean(cross_val_score(best_model, train_features, train_labels, cv=skf, scoring='recall_macro'))
            f1 = np.mean(cross_val_score(best_model, train_features, train_labels, cv=skf, scoring='f1_macro'))

            # Armazenar resultados
            resultados_finais['Modelo'].append(m)
            resultados_finais['Técnica'].append(nome)
            resultados_finais['Random State'].append(random_state)
            resultados_finais['Acurácia Treino'].append(acuracia)
            resultados_finais['Precisão Treino'].append(precisao)
            resultados_finais['Recall Treino'].append(recall)
            resultados_finais['F1-Score Treino'].append(f1)
            resultados_finais['Acurácia Teste'].append(acuracia_test)
            resultados_finais['Precisão Teste'].append(precisao_test)
            resultados_finais['Recall Teste'].append(recall_test)
            resultados_finais['F1-Score Teste'].append(f1_test)
            #resultados_finais['AUC-ROC'].append(auc_roc)


      # Criar DataFrame consolidado
      df_resultados = pd.DataFrame(resultados_finais)

      # Agrupar por Modelo e Técnica para calcular média e desvio padrão
      base_metric = 'Recall'

      # Monta os nomes das colunas dinamicamente
      treino_col = f"{base_metric} Treino"
      teste_col = f"{base_metric} Teste"

      # Agrupa e calcula médias
      df_resumo = df_resultados.groupby(['Modelo', 'Técnica']).agg(
          {treino_col: 'mean', teste_col: 'mean'}
          ).reset_index()

      # Exibe resultado ordenado pela métrica de teste (pode ajustar se quiser)
      print(f"\nDesempenho médio por modelo e técnica - {base_metric}:")
      df_resumo = df_resumo.sort_values(by=teste_col, ascending=False)
      print(df_resumo)
