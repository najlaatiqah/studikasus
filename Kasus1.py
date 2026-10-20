import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report

# ==========================================
# 1. SIMULASI DATASET (STUDI KASUS)
# ==========================================
np.random.seed(42)
n_samples = 4000

# Membuat data transaksi dengan tingkat keterlambatan sekitar 13%
data = {
    'customer_id': np.random.randint(101, 250, size=n_samples),
    'payment_method': np.random.choice(['Tempo 30 Hari', 'Tempo 60 Hari', 'Cash'], size=n_samples, p=[0.5, 0.4, 0.1]),
    'credit_limit': np.random.choice([5000000, 10000000, 20000000], size=n_samples),
    'order_value': np.random.randint(500000, 15000000, size=n_samples),
    'flag_late_payment': np.random.choice([0, 1], size=n_samples, p=[0.87, 0.13]) # Target: telat > 30 hari
}

df = pd.DataFrame(data)
print(f"Total baris data awal: {len(df)}")

# ==========================================
# 2. PENCEGAHAN DATA LEAKAGE (DATA PREPARATION)
# ==========================================
# Aturan: Fitur yang baru diketahui setelah peristiwa terjadi (seperti tanggal pembayaran aktual, 
# jumlah hari overdue, atau status akhir) harus dibuang agar model valid di dunia nyata.
# Pada script ini, kolom-kolom tersebut sengaja tidak disertakan ke dalam fitur (X).

features = ['payment_method', 'credit_limit', 'order_value']
X = pd.get_dummies(df[features], drop_first=True)
y = df['flag_late_payment']

# Membagi data latih dan data uji (Train-Test Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. PEMODELAN (MODELING)
# ==========================================
# Menggunakan Logistic Regression untuk memprediksi keterlambatan
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# 4. EVALUASI MODEL (EVALUATION)
# ==========================================
# Memeriksa metrik: Akurasi, Recall, dan ROC-AUC[cite: 3]
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n--- HASIL EVALUASI MODEL ---")
print(f"Akurasi Model : {acc * 100:.2f}% (Terlihat tinggi, namun menipu karena data tidak seimbang)[cite: 3]")
print(f"Recall Model  : {recall:.2f} (Model hampir tidak menemukan kasus telat bayar)[cite: 3]")
print(f"ROC-AUC Score : {roc_auc:.2f} (Berada di kisaran 0.6 yang menandakan model belum layak pakai)[cite: 3]")

print("\nLaporan Klasifikasi Lengkap:")
print(classification_report(y_test, y_pred, zero_division=0))