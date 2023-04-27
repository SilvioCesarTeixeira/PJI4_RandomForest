import pandas as pd
import pickle

# Carregar o modelo treinado
with open('modelo_random_forest.pkl', 'rb') as file:
    model = pickle.load(file)

# Carregar os dados de teste
data_test = pd.read_excel('venv/Realizar_Predicao.xlsx')  # Substitua pelo nome do seu arquivo de dados de teste

# Separar as features (X) e o target (y)
X_test = data_test

# Fazer a predição com o modelo
y_pred = model.predict(X_test)

# Comparar as predições com os valores reais
df_results = pd.DataFrame({'Predito': y_pred})
print(df_results)
