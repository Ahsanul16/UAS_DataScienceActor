import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- KONFIGURASI ---
DATA_PATH = os.path.join('data', 'processed', 'actors_clean.csv')
MODEL_PATH = os.path.join('models')
os.makedirs(MODEL_PATH, exist_ok=True)

def perform_eda(df):
    """Melakukan Exploratory Data Analysis dan menyimpan plot."""
    print("\n--- Memulai EDA ---")
    
    # Visualisasi 1: Distribusi Target (Lead vs Non-Lead)
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_lead', data=df)
    plt.title('Distribusi Peran (0=Support, 1=Lead)')
    plt.savefig('eda_distribution_role.png')
    print("Visualisasi 1 disimpan: eda_distribution_role.png")

    # Visualisasi 2: Gender vs Role
    plt.figure(figsize=(6, 4))
    sns.countplot(x='gender', hue='is_lead', data=df)
    plt.title('Perbandingan Peran berdasarkan Gender')
    plt.savefig('eda_gender_role.png')
    print("Visualisasi 2 disimpan: eda_gender_role.png")
    
    # Visualisasi 3: Top 10 Origin Aktor
    top_origins = df['origin'].value_counts().head(10).index
    plt.figure(figsize=(10, 6))
    sns.countplot(y='origin', data=df[df['origin'].isin(top_origins)], order=top_origins)
    plt.title('Top 10 Negara Asal Aktor')
    plt.savefig('eda_origin.png')
    print("Visualisasi 3 disimpan: eda_origin.png")

def train_models():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print("Dataset clean belum ada. Jalankan preprocessing.py dulu!")
        return

    df = pd.read_csv(DATA_PATH)
    perform_eda(df)

    # 2. Preprocessing untuk Modeling
    # Encode Gender dan Origin ke angka
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    
    le_origin = LabelEncoder()
    df['origin'] = le_origin.fit_transform(df['origin'])

    X = df[['gender', 'origin']]
    y = df['is_lead']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling (penting untuk Neural Network)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- MODEL 1: LOGISTIC REGRESSION (Baseline) ---
    print("\nTraining Model 1: Logistic Regression...")
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("Evaluasi Logistic Regression:")
    print(classification_report(y_test, y_pred_lr))

    # --- MODEL 2: RANDOM FOREST ---
    print("\nTraining Model 2: Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("Evaluasi Random Forest:")
    print(classification_report(y_test, y_pred_rf))

    # --- MODEL 3: DEEP LEARNING (Neural Network) ---
    print("\nTraining Model 3: Deep Learning (Neural Network)...")
    model = Sequential([
        Dense(16, input_dim=X_train.shape[1], activation='relu'),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid') # Binary classification
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Training dengan verbose agar log terlihat
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    
    # Evaluasi DL
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nEvaluasi Deep Learning -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Simpan Model DL
    model.save(os.path.join(MODEL_PATH, 'actor_role_model.h5'))
    print("Model Deep Learning berhasil disimpan.")

if __name__ == "__main__":
    train_models()