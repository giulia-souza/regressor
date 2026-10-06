import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

caminho = './datasets/vict/10000v/data_treino.csv' 
df_dataset = pd.read_csv(caminho)

colunas_proibidas = ['gcs', 'avpu', 'tri', 'sobr']
X = df_dataset.drop(columns=colunas_proibidas)
y = df_dataset['sobr']

scaler = StandardScaler() 
X_norm = scaler.fit_transform(X)#precisa ser normalizado p funcionar usando o sgd!!


validCruzada = KFold(n_splits=5, shuffle=True, random_state=42)

modelo_rn = MLPRegressor(
    max_iter=800, 
    alpha=0.0, 
    tol=1e-8, 
    n_iter_no_change=999, 
    random_state=42
)

parametros_rn = {
    'hidden_layer_sizes': [(1,), (5,), (100, 100, 100)],
    'activation': ['relu'],
    'solver': ['sgd'],
    'learning_rate_init': [0.03]
}

grid_rn = GridSearchCV(modelo_rn, parametros_rn, cv=validCruzada, scoring='neg_mean_squared_error', return_train_score=True)

print("\nA treinar as Redes Neurais Regressoras... (pode demorar alguns minutos)")
grid_rn.fit(X_norm, y)

resultados_rn = pd.DataFrame(grid_rn.cv_results_)
modelos_nomes_rn = ['U (1)', 'E (5)', 'O (100, 100, 100)']

for i, nome in enumerate(modelos_nomes_rn):
    print(f"\nMODELO {nome}:")
    
    # Inverter o sinal negativo do sklearn para obter o MSE real
    mse_treino_folds = -np.array([resultados_rn.loc[i, f'split{k}_train_score'] for k in range(5)])
    mse_val_folds = -np.array([resultados_rn.loc[i, f'split{k}_test_score'] for k in range(5)])
    
    diferencas_abs = np.abs(mse_treino_folds - mse_val_folds)
    
    media_treino = np.mean(mse_treino_folds)
    media_val = np.mean(mse_val_folds)
    media_diff = np.mean(diferencas_abs)
    
    dpa_treino = np.std(mse_treino_folds, ddof=1)
    dpa_val = np.std(mse_val_folds, ddof=1)
    dpa_diff = np.std(diferencas_abs, ddof=1)
    
    print(f"TREINO MSE (DPA)      | {media_treino:.5f} ({dpa_treino:.5f})")
    print(f"VALIDAÇÃO MSE (DPA)   | {media_val:.5f} ({dpa_val:.5f})")
    print(f"MÉDIA DAS DIFS. (DPA) | {media_diff:.5f} ({dpa_diff:.5f})")