import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def simulate_deforestation_classification():
    """
    Simulasi klasifikasi penyebab deforestasi hutan tropis menggunakan fitur citra satelit 
    (Sentinel-1 radar dan Sentinel-2 optik) berdasarkan materi pemantauan hutan.
    """
    print("--- SIMULASI KLASIFIKASI PENYEBAB DEFORESTASI DENGAN AI ---")
    
    # Simulasi data ekstraksi fitur citra satelit
    np.random.seed(42)
    n_samples = 1200
    
    data = {
        'optical_ndvi': np.random.uniform(0.1, 0.8, size=n_samples), # Indeks vegetasi dari Sentinel-2
        'radar_backscatter': np.random.uniform(-25.0, -5.0, size=n_samples), # Pantulan radar dari Sentinel-1
        'texture_variance': np.random.uniform(1.0, 50.0, size=n_samples), # Variasi spasial
        'near_road_distance': np.random.uniform(0.1, 15.0, size=n_samples), # Jarak ke jalan (km)
    }
    
    df = pd.DataFrame(data)
    
    # Simulasi label penyebab deforestasi:
    # 0: Pertanian, 1: Penebangan Kayu Selektif, 2: Pertambangan, 3: Kebakaran/Gangguan Alami
    conditions = [
        (df['near_road_distance'] < 3.0) & (df['optical_ndvi'] > 0.4),
        (df['radar_backscatter'] > -12.0),
        (df['texture_variance'] > 35.0),
    ]
    choices = [0, 1, 2]
    df['driver_cause'] = np.select(conditions, choices, default=3)
    
    print(f"Total sampel data pelatihan: {len(df)}")
    print(df.head())
    print("\n" + "="*50 + "\n")
    
    # Memisahkan Fitur (X) dan Target (y)
    X = df[['optical_ndvi', 'radar_backscatter', 'texture_variance', 'near_road_distance']]
    y = df['driver_cause']
    
    # Membagi data latih dan data uji (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pelatihan model (menggunakan Random Forest sebagai representasi pembelajaran pola)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("--- HASIL EVALUASI MODEL AI ---")
    print(f"Akurasi Model : {acc * 100:.2f}%")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=['Pertanian', 'Penebangan Selektif', 'Pertambangan', 'Kebakaran/Alami']))

if __name__ == "__main__":
    simulate_deforestation_classification()