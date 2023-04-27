import pandas as pd
import pickle

# Carregar o modelo treinado
with open('modelo_random_forest.pkl', 'rb') as file:
    model = pickle.load(file)

# Carregar os dados de teste
data_test = pd.read_excel('/home/silvio/PycharmProjects/PJI4_RandomForest/venv/Random_test.xlsx')  # Substitua pelo nome do seu arquivo de dados de teste

# Separar as features (X) e o target (y)
X_test = data_test.drop('VM_SRAG', axis=1)
y_test = data_test['VM_SRAG']

# Fazer a predição com o modelo
y_pred = model.predict(X_test)

# Comparar as predições com os valores reais
df_results = pd.DataFrame({'Real': y_test, 'Predito': y_pred})
print(df_results)
