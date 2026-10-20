import pandas as pd
import numpy as np

def preprocess_and_calibrate_sensor_data():
    """
    Simulasi pembersihan dan kalibrasi data sensor IoT (seperti PM2.5 dan cuaca)
    sebelum dijalankan pada infrastruktur HPC.
    Sesuai prinsip: Sensor perlu dikalibrasi sesuai kondisi lokal, 
    dan data dari berbagai sumber harus dibersihkan lebih dulu.
    """
    print("--- TAHAP PERSIAPAN DATA (DATA PREPARATION) ---")
    
    # Simulasi data mentah dari berbagai sensor IoT lokal yang berantakan
    raw_data = {
        'sensor_id': ['S_Kobe_01', 'S_Kobe_01', 'S_Kobe_02', 'S_Kobe_03', 'S_Kobe_02'],
        'pm25_raw': [55.2, -999.0, 62.5, 48.0, np.nan], # Mengandung nilai anomali/missing
        'wind_speed_ms': [3.2, 3.1, -1.0, 4.0, 2.8],      # Nilai kecepatan angin mustahil (-1.0)
        'temperature_c': [29.5, 29.6, 31.0, 28.2, 30.1]
    }
    
    df = pd.DataFrame(raw_data)
    print("Data Mentah Awal:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # 1. Menangani anomali dan nilai hilang (Missing Values)
    # Mengubah nilai error/anomali ekstrem (-999.0) menjadi NaN
    df['pm25_clean'] = df['pm25_raw'].replace(-999.0, np.nan)
    df['wind_clean'] = df['wind_speed_ms'].apply(lambda x: np.nan if x < 0 else x)
    
    # Mengisi missing value dengan rata-rata dari masing-masing kolom
    df['pm25_clean'] = df['pm25_clean'].fillna(df['pm25_clean'].mean())
    df['wind_clean'] = df['wind_clean'].fillna(df['wind_clean'].mean())
    
    # 2. Kalibrasi Sensor Sesuai Kondisi Lokal
    # Faktor koreksi kalibrasi lapangan terhadap pembacaan sensor fisik
    calibration_factor_pm25 = 1.08 
    df['pm25_calibrated'] = df['pm25_clean'] * calibration_factor_pm25
    
    # Menyeleksi kolom akhir yang bersih untuk dikirim ke HPC
    final_features = df[['sensor_id', 'pm25_calibrated', 'wind_clean', 'temperature_c']]
    
    print("Data Setelah Dibersihkan dan Dikalibrasi (Siap untuk Batch Job HPC):")
    print(final_features)
    
    # Menyimpan hasil pembersihan ke format file (misal: CSV) untuk di-upload ke server HPC via SSH
    # final_features.to_csv("clean_sensor_data_hpc.csv", index=False)
    return final_features

if __name__ == "__main__":
    cleaned_dataset = preprocess_and_calibrate_sensor_data()