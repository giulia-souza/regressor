import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


print("\nA iniciar o Teste Cego...")

caminho_cego = './datasets/vict/1300v/data_testeCego.csv' 
df_cego = pd.read_csv(caminho_cego)

# Preparar dados: separar o alvo e remover as colunas proibidas
X_cego = df_cego.drop(columns=['gcs', 'avpu', 'tri', 'sobr'])
y_cego = df_cego['sobr']

scaler = joblib.load('scaler.joblib')
melhor_cart = joblib.load('melhor_cart.joblib')
melhor_rn = joblib.load('melhor_rn.joblib')

X_cego_norm = scaler.transform(X_cego)

y_pred_cart = melhor_cart.predict(X_cego)
y_pred_rn = melhor_rn.predict(X_cego_norm)

# Calcular MSE e RMSE
mse_cart = mean_squared_error(y_cego, y_pred_cart)
rmse_cart = np.sqrt(mse_cart)

mse_rn = mean_squared_error(y_cego, y_pred_rn)
rmse_rn = np.sqrt(mse_rn)

print(f"Melhor CART -> MSE: {mse_cart:.5f} | RMSE: {rmse_cart:.5f}")
print(f"Melhor RN   -> MSE: {mse_rn:.5f} | RMSE: {rmse_rn:.5f}")

# Calcular o Ganho %
if mse_cart < mse_rn:
    ganho_mse = (1 - (mse_cart / mse_rn)) * 100
    ganho_rmse = (1 - (rmse_cart / rmse_rn)) * 100
    print(f"\nVENCEDOR DO TESTE CEGO: CART")
    print(f"Ganho % de MSE: {ganho_mse:.2f}%")
    print(f"Ganho % de RMSE: {ganho_rmse:.2f}%")
else:
    ganho_mse = (1 - (mse_rn / mse_cart)) * 100
    ganho_rmse = (1 - (rmse_rn / rmse_cart)) * 100
    print(f"\nVENCEDOR DO TESTE CEGO: REDE NEURAL")
    print(f"Ganho % de MSE: {ganho_mse:.2f}%")
    print(f"Ganho % de RMSE: {ganho_rmse:.2f}%")

plt.figure(figsize=(12, 5))

# Gráfico CART
plt.subplot(1, 2, 1)
plt.scatter(y_cego, y_pred_cart, alpha=0.5, color='blue')
plt.plot([0, 1], [0, 1], '--', color='red', linewidth=2) # A linha da perfeição
plt.title('CART: Real vs Predito')
plt.xlabel('Probabilidade Real (sobr)')
plt.ylabel('Probabilidade Predita')
plt.grid(True)

# Gráfico RN
plt.subplot(1, 2, 2)
plt.scatter(y_cego, y_pred_rn, alpha=0.5, color='green')
plt.plot([0, 1], [0, 1], '--', color='red', linewidth=2) # A linha da perfeição
plt.title('Rede Neural: Real vs Predito')
plt.xlabel('Probabilidade Real (sobr)')
plt.ylabel('Probabilidade Predita')
plt.grid(True)

plt.tight_layout()
plt.show()