import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)

#sns.countplot(y='NObeyesdad',data=data)
#plt.show()

#print(data.isnull().sum()) #soma as quantidades nulas de cada coluna

#print(data.info()) #informa as colunas e o tipo de dado
#print(data.describe()) #descreve os dados numéricos

colun_continuos = data.select_dtypes(include=['float64']).columns.tolist() #põe o rótulo das colunas com dados numéricos

#print(data[colun_continuos])

scaler = StandardScaler() 
features = scaler.fit_transform(data[colun_continuos]) #normaliza os dados (mu=0, std=1)

scaled_df = pd.DataFrame(features,columns=scaler.get_feature_names_out(colun_continuos)) #novo data frame
scaled_data = pd.concat([data.drop(columns=colun_continuos),scaled_df],axis=1)


categorical = scaled_data.select_dtypes(include=['object']).columns.tolist() #pega o rótulo das colunas com dados categorizados
categorical.remove('NObeyesdad')

#print(data[categorical])

encoder = OneHotEncoder(sparse_output=False,drop='first')
encoder_feat = encoder.fit_transform(scaled_data[categorical])

encoder_df = pd.DataFrame(encoder_feat,columns=encoder.get_feature_names_out(categorical)) #novo dataframe
encoder_data = pd.concat([scaled_data.drop(columns=categorical),encoder_df],axis=1) #Junta todas as colunas (numerico+categórico)

#print(encoder_data)

encoder_data['NObeyesdad'] = encoder_data['NObeyesdad'].astype('category').cat.codes #categoriza por numeros inteiros

X = encoder_data.drop('NObeyesdad',axis=1)
y = encoder_data['NObeyesdad']

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.33,random_state=42) #separa conjuntos de treino(2/3) e teste(1/3)

model_ova = LogisticRegression(multi_class='ovr',max_iter=1000) #treino de regressão logistica One-vs-All
model_ova.fit(xtrain,ytrain)

y_pred = model_ova.predict(xtest)

print(accuracy_score(ytest,y_pred))

model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000)) #treino de regressão logistica One-vs-one
model_ovo.fit(xtrain,ytrain)

yovo_pred = model_ovo.predict(xtest)

print(accuracy_score(ytest,yovo_pred))