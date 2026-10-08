import joblib
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

caminho = './datasets/vict/10000v/data_treino.csv' 
df_dataset = pd.read_csv(caminho)

colunas_proibidas = ['gcs', 'avpu', 'tri', 'sobr']
X = df_dataset.drop(columns=colunas_proibidas)
y = df_dataset['sobr']

scaler = StandardScaler() 
X_norm = scaler.fit_transform(X)#precisa ser normalizado p funcionar usando o sgd!!


melhor_cart = DecisionTreeRegressor(max_depth=8, random_state=42)
melhor_rn = MLPRegressor(
    hidden_layer_sizes=(5,),       
    activation='relu', 
    solver='sgd', 
    learning_rate_init=0.03,
    max_iter=800,
    alpha=0.0,
    tol=1e-8,
    n_iter_no_change=999,
    random_state=42
)

melhor_cart.fit(X, y)
melhor_rn.fit(X_norm, y)

joblib.dump(melhor_cart, 'melhor_cart.joblib')
joblib.dump(melhor_rn, 'melhor_rn.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Ficheiros .joblib guardados com sucesso!")

y_pred_treino_rn = melhor_rn.predict(X_norm)
y_pred_treino_cart = melhor_cart.predict(X)

mse_treino_total_rn = mean_squared_error(y, y_pred_treino_rn)
mse_treino_total_cart = mean_squared_error(y, y_pred_treino_cart)

print(f"MSE de Treino absoluto nas 10.000 vítimas (CART E): {mse_treino_total_cart:.5f}")
print(f"MSE de Treino absoluto nas 10.000 vítimas (RN E):   {mse_treino_total_rn:.5f}")
    
