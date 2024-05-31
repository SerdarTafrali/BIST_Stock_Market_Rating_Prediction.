# Veri Yükleme ve Ön İşleme

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Veri setini yükleme
file_path = '/mnt/data/Bist Screener_2024-05-31.csv'
data = pd.read_csv(file_path)

# Eksik verileri temizleme
data = data.dropna()

# 'Analyst Rating' değişkenini sayısal değere dönüştürme
label_encoder = LabelEncoder()
data['Analyst Rating'] = label_encoder.fit_transform(data['Analyst Rating'])

# Özellikleri ve hedef değişkeni ayırma
X = data.drop(columns=['Analyst Rating', 'Symbol', 'Description', 'Price - Currency', 
                       'Target price 1 year - Currency', 'Technical Rating 1 week'])
y = data['Analyst Rating']

# Sayısal değişkenleri ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Optuna ile Hiperparametre Optimizasyonu
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# Optuna ile LSTM model hiperparametrelerini optimize etme
def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    units = trial.suggest_int('units', 10, 100)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout_rate))
    for _ in range(n_layers - 1):
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout_rate))
    model.add(LSTM(units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=0)
    
    accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)[1]
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# En iyi hiperparametreleri gösterme
best_params = study.best_params
best_params

# En İyi Parametrelerle Model Eğitimi ve Değerlendirme
# Optuna ile bulunan en iyi parametreler
best_params = study.best_params
n_layers = best_params['n_layers']
units = best_params['units']
dropout_rate = best_params['dropout_rate']
learning_rate = best_params['learning_rate']

# En iyi parametrelerle modeli oluşturma
model = Sequential()
model.add(LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(dropout_rate))
for _ in range(n_layers - 1):
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(dropout_rate))
model.add(LSTM(units))
model.add(Dropout(dropout_rate))
model.add(Dense(1, activation='sigmoid'))

# Modeli derleme
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Veriyi LSTM modeline uygun hale getirme
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Modeli eğitme
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Modelin performansını değerlendirme
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
loss, accuracy

