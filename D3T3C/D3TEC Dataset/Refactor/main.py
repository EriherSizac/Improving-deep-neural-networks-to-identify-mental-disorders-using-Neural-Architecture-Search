import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from utils import encode_model_architecture, fixArch, decode_model_architecture, build_tf_model_from_dict

# Cargar datos
data = pd.read_csv('./model_results_combined.csv')
data['Encoded Architecture'] = data['Encoded Architecture'].apply(eval)
X = pd.DataFrame(data['Encoded Architecture'].tolist())
y_metrics = data[['Loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'Specificity']]

X_train, X_test, y_train, y_test = train_test_split(X, y_metrics, test_size=0.2, random_state=42)

# Optimización de hiperparámetros para modelos surrogados
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

for metric in y_metrics.columns:
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train[metric])
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f"{metric}_best_model.pkl")
    print(f"{metric}: Mejor MSE en validación: {-grid_search.best_score_}")

# Predicción de arquitecturas aleatorias
for i in range(10):
    random_architecture = encode_model_architecture(decode_model_architecture(fixArch(np.random.randint(0, 8, 48))))
    example_architecture = np.array(random_architecture).reshape(1, -1)
    for metric in y_metrics.columns:
        model = joblib.load(f"{metric}_best_model.pkl")
        print(f"Predicción para {metric}: {model.predict(example_architecture)[0]}")
