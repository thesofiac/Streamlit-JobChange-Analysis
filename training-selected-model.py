import joblib
import pandas as pd
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

  features = df.drop(columns=['target','city','enrollee_id']).astype(float)
  labels = df['target']

  return features, labels


# Carregar modelo
pipeline = joblib.load('final_model.pkl')

# Tratar colunas com a MESMA função usada no treino
features, labels = treat_columns(df_train)

# Fazer predição
preds = pipeline.predict(features)
recall_train = recall_score(labels, preds, average='macro')
print("Recall: ", recall_train)

