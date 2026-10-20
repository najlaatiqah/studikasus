import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def predict_house_prices():
    """
    Simulasi studi kasus Machine Learning untuk prediksi harga rumah (Regresi).
    Berdasarkan materi Introduction to Machine Learning[cite: 19, 22].
    """
    print("--- TAHAP 1 & 2: SIMULASI DATA & PRA-PEMROSESAN ---")
    
    # Simulasi dataset historis rumah
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'luas_bangunan_m2': np.random.randint(45, 300, size=n_samples),
        'kamar_tidur': np.random.randint(1, 6, size=n_samples),
        'kamar_mandi': np.random.randint(1, 4, size=n_samples),
        'lokasi_kategori': np.random.choice([1, 2, 3], size=n_samples, p=[0.5, 0.3, 0.2]), # 1: Pinggiran, 2: Sub-urban, 3: Perkotaan
    }
    
    # Simulasi target harga berdasarkan formula linier ditambah sedikit noise/random
    df = pd.DataFrame(data)
    df['harga_juta_rp'] = (
        (df['luas_bangunan_m2'] * 12.5) + 
        (df['kamar_tidur'] * 45) + 
        (df['kamar_mandi'] * 30) + 
        (df['lokasi_kategori'] * 150) + 
        np.random.normal(0, 20, size=n_samples)
    )
    
    print(f"Total baris data: {len(df)}")
    print(df.head())
    print("\n" + "="*50 + "\n")
    
    # Menentukan Variabel Fitur (X) dan Target (y)
    X = df[['luas_bangunan_m2', 'kamar_tidur', 'kamar_mandi', 'lokasi_kategori']]
    y = df['harga_juta_rp']
    
    # --- TAHAP 3: PEMBAGIAN DATA (Train-Test Split 80:20) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- TAHAP 4: PEMILIHAN & PELATIHAN MODEL (Model Training) ---
    # Menggunakan Linear Regression karena target berupa nilai kontinu (angka)[cite: 21]
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # --- TAHAP 5: EVALUASI MODEL ---
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("--- HASIL EVALUASI MODEL REGRESI ---")
    print(f"Mean Absolute Error (MAE) : {mae:.2f} (Rata-rata selisih prediksi harga)")
    print(f"R-squared (R2 Score)      : {r2:.4f} (Seberapa baik model menjelaskan variasi data)")
    
    # Simulasi prediksi untuk rumah baru
    print("\n--- SIMULASI PREDIKSI RUMAH BARU ---")
    rumah_baru = pd.DataFrame([[120, 3, 2, 3]], columns=['luas_bangunan_m2', 'kamar_tidur', 'kamar_mandi', 'lokasi_kategori'])
    prediksi_harga = model.predict(rumah_baru)
    print(f"Estimasi Harga untuk Rumah (120m2, 3 KT, 2 KM, Lokasi Perkotaan): Rp {prediksi_harga[0]:.2f} juta")

if __name__ == "__main__":
    predict_house_prices()