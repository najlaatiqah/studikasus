import pandas as pd
import plotly.express as px

def create_interactive_dashboard():
    """
    Simulasi pembuatan visualisasi data interaktif untuk dasbor e-commerce.
    Menerapkan prinsip pemilihan grafik yang tepat dan minim chartjunk[cite: 30].
    """
    print("--- MEMPROSES DATA UNTUK DASBOR IMK & VISUALISASI ---")
    
    # Simulasi data penjualan bulanan berdasarkan kategori produk
    data = {
        'Bulan': ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun',
                  'Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun'],
        'Kategori': ['Elektronik', 'Elektronik', 'Elektronik', 'Elektronik', 'Elektronik', 'Elektronik',
                     'Pakaian', 'Pakaian', 'Pakaian', 'Pakaian', 'Pakaian', 'Pakaian'],
        'Pendapatan_Juta': [120, 135, 150, 140, 170, 190, 80, 95, 90, 110, 125, 140]
    }
    
    df = pd.DataFrame(data)
    
    # 1. Line Chart untuk melihat perubahan tren seiring waktu (Sesuai panduan materi)[cite: 30]
    fig_line = px.line(
        df, 
        x='Bulan', 
        y='Pendapatan_Juta', 
        color='Kategori',
        markers=True,
        title='Tren Pendapatan Bulanan Berdasarkan Kategori (Overview)'
    )
    
    # Menyesuaikan tata letak agar bersih dan minimalis (mengurangi beban kognitif)
    fig_line.update_layout(
        plot_bgcolor='white',
        font=dict(family="Arial", size=12),
        title_font=dict(size=16, color='black')
    )
    
    print("Grafik tren berhasil disiapkan sesuai prinsip visualisasi data.")
    print("Menampilkan ringkasan data awal:")
    print(df.head())
    
    # Untuk menjalankan di lingkungan lokal Streamlit/Dash, gunakan:
    # fig_line.show()
    
    return df, fig_line

if __name__ == "__main__":
    dataset, dashboard_figure = create_interactive_dashboard()