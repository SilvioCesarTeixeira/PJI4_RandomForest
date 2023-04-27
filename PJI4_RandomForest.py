import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Referenciar base de dados existente na rede
url_SG = r'https://github.com/SilvioCesarTeixeira/PJI4/raw/master/SRAG_SubP_SP.xlsx'
url_QA = r'https://github.com/SilvioCesarTeixeira/PJI4/raw/master/Santana_SP_AQI.xlsx'

# Criar Dataframes
df1 = pd.read_excel(url_QA, engine='openpyxl')
df2 = pd.read_excel(url_SG, engine='openpyxl')

# Definir a linha zero do dataframe 1 como título das colunas
df1.columns = df1.iloc[0]
df1 = df1.reindex(df1.index.drop(0))

# Definindo a primeira coluna como data com padrão português do Brasil
df_QA = df1
df_QA['Data'] = pd.to_datetime(df1['date'], format='%Y/%m/%d').dt.strftime('%d/%m/%Y')
col_Data_Brasil = df_QA.iloc[:, -1]
df_QA = df_QA.iloc[:, 1:-1]
df_QA.insert(loc=0, column='Data', value=col_Data_Brasil)

# Remover linhas desnecessárias e renomear coluna do dataframe 2
df_SRAG = df2
df_SRAG.columns = df_SRAG.iloc[2]
linhas_remover = list(range(0, 3)) + list(range(1164, 1168))
df_SRAG = df_SRAG.reindex(df_SRAG.index.drop(linhas_remover))
df_SRAG = df_SRAG.rename(columns={'Ano M�s Dia Notifica��o': 'Data'})

# Juntar os dois dataframes, mas mantendo apenas as datas que coincidem
df_SRAG = df_SRAG.loc[:, ['Data', 'VILA MARIA/VILA GUILHERME']]
df_base = pd.merge(df_QA, df_SRAG, how='inner', on='Data')

# Definir uma data de corte
data_limite = '01/06/2021'
df_base['Data'] = pd.to_datetime(df_base['Data'], format='%d/%m/%Y')
df_base = df_base.loc[df_base['Data'] > data_limite]

# Renomear colunas com título extenso e converter strings em tipos numéricos
df_base = df_base.rename(columns={' pm25': 'PM25', ' o3': 'O3', 'VILA MARIA/VILA GUILHERME': 'VM_SRAG'})
df_base['PM25'] = pd.to_numeric(df_base['PM25'], errors='coerce')
df_base[' pm10'] = pd.to_numeric(df_base[' pm10'], errors='coerce')
df_base['O3'] = pd.to_numeric(df_base['O3'], errors='coerce')
df_base[' no2'] = pd.to_numeric(df_base[' no2'], errors='coerce')
df_base['VM_SRAG'] = pd.to_numeric(df_base['VM_SRAG'], errors='coerce')

# Definir um dataframe base apenas com as colunas que contenham dados em quase todas as linhas
df_base = df_base.loc[:, ['Data', 'PM25', 'O3', 'VM_SRAG']]

# Preencher os valores NaN (não numéricos) com a média de cada variável
df_base['PM25'].fillna(value=df_base['PM25'].mean(), inplace=True)
df_base['O3'].fillna(value=df_base['O3'].mean(), inplace=True)
df_base['VM_SRAG'].fillna(value=df_base['VM_SRAG'].mean(), inplace=True)

# Criar dataframe para calcular correlações entre variáveis
df_corr = df_base
df_corr['Dias'] = (pd.to_datetime(df_corr['Data']) - pd.to_datetime('1900-01-01')).dt.days
df_corr = df_corr.drop(['Data'], axis=1)
df_corr.info()

# Gerar uma matriz de correlação entre as variáveis do dataframe base
corr_matriz1 = df_corr.corr()

# Plotar o Mapa de Calor com as correlações calculadas
sns.heatmap(corr_matriz1, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlação entre as variáveis QA-SRAG')
plt.xlabel('Variáveis')
plt.ylabel('Variáveis')
plt.show()

# Plotar gráfico com dados do dataframe ao longo dos anos de 2020 a 2023.
df_base['Data'] = pd.to_datetime(df_base['Data'], format='%d/%m/%Y')

df_base = df_base.sort_values(['Data'], ascending=True)
print(df_base)

plt.figure(figsize=(20, 12))
plt.plot(df_base['Data'], df_base['PM25'], label='PM25')
plt.plot(df_base['Data'], df_base['O3'], label='O3')
plt.plot(df_base['Data'], df_base['VM_SRAG'], label='VM_SRAG')
plt.xlabel('Data')
plt.ylabel('Valor')
plt.legend()
plt.show()

df_base_mmAAAA = df_base.groupby([df_base['Data'].dt.year, df_base['Data'].dt.month])['VM_SRAG'].sum()
df_base_mmAAAA.plot(kind='line', figsize=(10, 5))
plt.title('Número de internações por mês')
plt.xlabel('Mês/Ano')
plt.ylabel('Internações')
plt.show()

# Utilizar dataframe
Y = df_base.drop(['Data', 'PM25', 'O3', 'Dias'], axis=1)

# Utilizar Dataframe
X = df_base.drop(['VM_SRAG', 'Data'], axis=1)

# Converter a coluna de datas para um formato adequado (se necessário)
# X['Data'] = pd.to_datetime(X['Data'])

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Instanciar e treinar o modelo Random Forest
model = RandomForestRegressor()
y_train = y_train['VM_SRAG'].values.ravel()
model.fit(X_train, y_train)

# Fazer previsões com o conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o desempenho do modelo usando a métrica de erro quadrático médio (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)

# Avaliar o desempenho do modelo usando a métrica de erro absoluto médio (MAE)
mae = mean_absolute_error(y_test, y_pred)
print('MAE', mae)

# Avaliar o desempenho do modelo usando a métrica de erro quadrático médio (R2)
r2 = r2_score(y_test, y_pred)
print('R2:', r2)

plt.plot(y_test, label='Valores Reais', marker='.', linestyle='None')
plt.plot(y_pred, label='Predição', marker='.', linestyle='None')
plt.legend()
plt.show()

with open('modelo_random_forest.pkl', 'wb') as file:
    pickle.dump(model, file)
