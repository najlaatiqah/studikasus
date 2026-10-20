import pandas as pd
import numpy as np

def analyze_urban_tree_assets():
    """
    Simulasi pemrosesan dan analisis data aset spasial pohon perkotaan 
    sebelum diunggah ke database PostGIS atau platform cloud GIS.
    Mencerminkan peran lulusan Informatika sebagai pengembang tools GIS.
    """
    print("--- TAHAP PEMROSESAN DATA SPASIAL ASET KOTA ---")
    
    # Simulasi data mentah laporan aset pohon dari lapangan (Geo-tagged data)
    raw_data = {
        'asset_id': ['TR-001', 'TR-002', 'TR-003', 'TR-004', 'TR-005'],
        'latitude': [-43.5321, -43.5512, np.nan, -43.5110, -43.5422],
        'longitude': [172.6362, 172.5831, 172.6120, np.nan, 172.6011],
        'tree_height_m': [12.5, 18.0, 8.5, 22.1, 14.3],
        'health_status': ['Sehat', 'Berisiko Tumbang', 'Sehat', 'Berisiko Tumbang', 'Perlu Pemangkasan']
    }
    
    df = pd.DataFrame(raw_data)
    print("Data Mentah Awal dari Lapangan:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # 1. Pembersihan Data (Data Cleaning): Membuang data dengan koordinat yang hilang
    df_clean = df.dropna(subset=['latitude', 'longitude']).copy()
    
    # 2. Analisis Berbasis Aturan (Rule-based Filtering untuk Prioritas Pemeliharaan)
    # Menandai pohon dengan tinggi > 15 meter dan status berisiko sebagai prioritas tinggi penanganan
    df_clean['high_priority_action'] = df_clean.apply(
        lambda row: 'Ya (Segera Tangani)' if row['tree_height_m'] > 15 and 'Berisiko' in row['health_status'] else 'Normal', 
        axis=1
    )
    
    print("Data Setelah Dibersihkan dan Dianalisis untuk Alokasi Anggaran Perbaikan:")
    print(df_clean[['asset_id', 'latitude', 'longitude', 'tree_height_m', 'high_priority_action']])
    
    # Menyimpan hasil pemrosesan ke format file (misal: CSV atau GeoJSON) untuk di-load ke QGIS / PostGIS
    # df_clean.to_csv("processed_tree_assets.csv", index=False)
    return df_clean

if __name__ == "__main__":
    processed_assets = analyze_urban_tree_assets()