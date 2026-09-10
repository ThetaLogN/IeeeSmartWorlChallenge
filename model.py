import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Caricamento del dataset
print("Caricamento dati...")
df = pd.read_csv('smart_city_traffic_mobility.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by=['intersection_id', 'timestamp']).reset_index(drop=True)

# 2. Feature Engineering: Lag features per ogni incrocio
print("Creazione Lag Features...")
df['vehicle_count_lag1'] = df.groupby('intersection_id')['vehicle_count'].shift(1)
df['average_speed_lag1'] = df.groupby('intersection_id')['average_speed'].shift(1)
df['queue_length_lag1'] = df.groupby('intersection_id')['queue_length'].shift(1)

# Feature cicliche per l'ora
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)

# Target: domanda veicolare futura al passo successivo (t+1)
df['target_future_vehicles'] = df.groupby('intersection_id')['vehicle_count'].shift(-1)

# Rimuove righe con NaN dovute allo shift
df = df.dropna(subset=['vehicle_count_lag1', 'target_future_vehicles']).reset_index(drop=True)

# 3. Definizione delle Feature
feature_cols = [
    # Memoria recente
    'vehicle_count_lag1', 'average_speed_lag1', 'queue_length_lag1',
    # Tempo e stagionalità
    'sin_hour', 'cos_hour', 'day_of_week', 'is_weekend', 'is_holiday', 'rush_hour',
    # Meteo e ambiente
    'temperature', 'rainfall_mm', 'visibility_km', 'air_quality_index',
    # Infrastruttura e POI
    'lanes', 'speed_limit', 'nearby_school', 'nearby_hospital', 'nearby_market',
    # Shock e anomalie
    'construction_activity', 'accident_reported', 'public_event', 'emergency_vehicle_detected'
]

X = df[feature_cols]
y = df['target_future_vehicles']

# 4. Split Temporale Cronologico (80% Train, 20% Test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Dimensione Training Set: {X_train.shape[0]} osservazioni")
print(f"Dimensione Test Set:     {X_test.shape[0]} osservazioni")

# 5. Addestramento del Modello
print("Addestramento Gradient Boosting Regressor in corso...")
model = HistGradientBoostingRegressor(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- RISULTATI DI PERFORMANCE (TEST SET CRONOLOGICO) ---")
print(f"R² Score (Bontà di adattamento): {r2:.4f}  (ideale > 0.90)")
print(f"MAE (Errore Medio Assoluto):     {mae:.2f} veicoli/ora")
print(f"RMSE (Root Mean Squared Error):  {rmse:.2f} veicoli/ora")
