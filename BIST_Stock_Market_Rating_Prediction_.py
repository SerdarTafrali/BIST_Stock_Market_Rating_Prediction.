# Veri Yükleme ve Ön İşleme

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Veri setini yükleme
file_path = 'C:/Users/SERDAR/Documents/Okul/Uskudar_Universitesi/Yapay_Zeka_ve_Uygulamaları/final/234329035_serdar_tafrali/BIST_Stock_Market_Rating_Prediction_Dataset.csv'
data = pd.read_csv(file_path)

# Özellikleri ve hedef değişkeni ayırma
X = data.drop(columns=['Analyst Rating', 'Symbol', 'Description', 'Price - Currency', 
                       'Target price 1 year - Currency', 'Technical Rating 1 week'])
y = data['Analyst Rating']

# Sayısal ve kategorik sütunları belirleme
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Ön işleme adımları: sayısal veriler için ölçeklendirme, kategorik veriler için one-hot encoding
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Ön işleme adımlarını birleştirme
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Ön işleme ve model adımlarını birleştirme
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi dönüştürme
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

# Optuna ile Hiperparametre Optimizasyonu

import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# Hedef değişkeni dönüştürme
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# NumPy dizilerine dönüştürme ve veri türünü kontrol etme
y_train_encoded = np.array(y_train_encoded).astype('float32')
y_test_encoded = np.array(y_test_encoded).astype('float32')

# Optuna ile LSTM model hiperparametrelerini optimize etme
def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    units = trial.suggest_int('units', 10, 100)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
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
    
    history = model.fit(X_train_reshaped, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test_encoded), verbose=0)
    
    accuracy = model.evaluate(X_test_reshaped, y_test_encoded, verbose=0)[1]
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

# En iyi hiperparametreleri gösterme
best_params = study.best_params
best_params

# En İyi Parametrelerle Model Eğitimi ve Değerlendirme
best_units = best_params['units']
best_dropout_rate = best_params['dropout_rate']
best_learning_rate = best_params['learning_rate']
best_n_layers = best_params['n_layers']

# En iyi hiperparametrelerle LSTM modelini oluşturma
best_model = Sequential()
best_model.add(LSTM(best_units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
best_model.add(Dropout(best_dropout_rate))
for _ in range(best_n_layers - 1):
    best_model.add(LSTM(best_units, return_sequences=True))
    best_model.add(Dropout(best_dropout_rate))
best_model.add(LSTM(best_units))
best_model.add(Dropout(best_dropout_rate))
best_model.add(Dense(1, activation='sigmoid'))

best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# Veriyi yeniden şekillendirme
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# En iyi model ile eğitim
history = best_model.fit(X_train_reshaped, y_train_encoded, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test_encoded), verbose=1)

# Modelin değerlendirilmesi
train_loss, train_accuracy = best_model.evaluate(X_train_reshaped, y_train_encoded, verbose=0)
test_loss, test_accuracy = best_model.evaluate(X_test_reshaped, y_test_encoded, verbose=0)

print(f'Training accuracy: {train_accuracy:.4f}')
print(f'Testing accuracy: {test_accuracy:.4f}')

# Eğitim ve doğrulama doğruluğu ve kaybının görselleştirilmesi
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
