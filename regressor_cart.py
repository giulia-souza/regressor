import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor

caminho = './datasets/vict/10000v/data_treino.csv' 
df_dataset = pd.read_csv(caminho)

colunas_proibidas = ['gcs', 'avpu', 'tri', 'sobr']
X = df_dataset.drop(columns=colunas_proibidas)
y = df_dataset['sobr']

validCruzada = KFold(n_splits=5, shuffle=True, random_state=42)

modelo_cart = DecisionTreeRegressor(random_state=42)

parametros = {'max_depth': [3, 8, None]} 

# O sklearn utiliza 'neg_mean_squared_error', que devolve o erro negativo (resolveremos isso no print)
grid = GridSearchCV(modelo_cart, parametros, cv=validCruzada, scoring='neg_mean_squared_error', return_train_score=True)

grid.fit(X, y)

resultados = pd.DataFrame(grid.cv_results_)
modelos_nomes = ['U (max_depth=3)', 'E (max_depth=8)', 'O (max_depth=None)']

for i, nome in enumerate(modelos_nomes):
    print(f"\nMODELO {nome}:")
    
    # mult por -1 para converter o erro negativo do sklearn no MSE real
    mse_treino_folds = -np.array([resultados.loc[i, f'split{k}_train_score'] for k in range(5)])
    mse_val_folds = -np.array([resultados.loc[i, f'split{k}_test_score'] for k in range(5)])
    
    diferencas_abs = np.abs(mse_treino_folds - mse_val_folds)
    
    media_treino = np.mean(mse_treino_folds)
    dpa_treino = np.std(mse_treino_folds, ddof=1)
    
    media_val = np.mean(mse_val_folds)
    dpa_val = np.std(mse_val_folds, ddof=1)
    
    media_diff = np.mean(diferencas_abs)
    dpa_diff = np.std(diferencas_abs, ddof=1)
    
    print(f"TREINO MSE (DPA)      | {media_treino:.5f} ({dpa_treino:.5f})")
    print(f"VALIDAÇÃO MSE (DPA)   | {media_val:.5f} ({dpa_val:.5f})")
    print(f"MÉDIA DAS DIFS. (DPA) | {media_diff:.5f} ({dpa_diff:.5f})")