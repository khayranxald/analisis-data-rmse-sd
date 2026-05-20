import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#========== CONTOH 1: Data Prediksi vs Aktual ==========
print("=" * 50)
print("CONTOH 1: Menghitung RMSE")
print("=" * 50)

# Data aktual dan prediksi
data_aktual = [100, 120, 130, 140, 150, 160, 170, 180, 190, 200]
data_prediksi = [98, 125, 128, 145, 148, 162, 168, 182, 188, 205]

# Cara 1: Manual
errors = []
for i in range(len(data_aktual)):
    error = data_aktual[i] - data_prediksi[i]
    errors.append(error)
    print(f"Data ke-{i+1}: Aktual={data_aktual[i]}, Prediksi={data_prediksi[i]}, Error={error}")

# Hitung RMSE manual
squared_errors = [e**2 for e in errors]
mean_squared_error_manual = sum(squared_errors) / len(squared_errors)
rmse_manual = mean_squared_error_manual ** 0.5

print(f"\nRMSE (Manual): {rmse_manual:.4f}")

# Cara 2: Menggunakan NumPy
rmse_numpy = np.sqrt(np.mean((np.array(data_aktual) - np.array(data_prediksi))**2))
print(f"RMSE (NumPy): {rmse_numpy:.4f}")

# Cara 3: Menggunakan sklearn
rmse_sklearn = np.sqrt(mean_squared_error(data_aktual, data_prediksi))
print(f"RMSE (sklearn): {rmse_sklearn:.4f}")

# ========== CONTOH 2: Menghitung Standard Deviation ==========
print("\n" + "=" * 50)
print("CONTOH 2: Menghitung Standard Deviation")
print("=" * 50)

# Data nilai siswa
nilai_siswa = [75, 80, 85, 70, 90, 88, 92, 78, 83, 87, 95, 72, 89, 91, 84]

print(f"Data nilai siswa: {nilai_siswa}")
print(f"Jumlah data: {len(nilai_siswa)}")

# Cara 1: Manual
mean = sum(nilai_siswa) / len(nilai_siswa)
print(f"\nMean (rata-rata): {mean:.2f}")

variance = sum((x - mean)**2 for x in nilai_siswa) / len(nilai_siswa)
std_dev_population = variance ** 0.5

print(f"Variance (populasi): {variance:.2f}")
print(f"Standard Deviation (populasi): {std_dev_population:.2f}")

# Standard deviation sampel (n-1)
variance_sample = sum((x - mean)**2 for x in nilai_siswa) / (len(nilai_siswa) - 1)
std_dev_sample = variance_sample ** 0.5
print(f"Standard Deviation (sampel): {std_dev_sample:.2f}")

# Cara 2: Menggunakan NumPy
std_numpy_pop = np.std(nilai_siswa)  # populasi (ddof=0)
std_numpy_sample = np.std(nilai_siswa, ddof=1)  # sampel (ddof=1)

print(f"\nStandard Deviation NumPy (populasi): {std_numpy_pop:.2f}")
print(f"Standard Deviation NumPy (sampel): {std_numpy_sample:.2f}")

# ========== CONTOH 3: Analisis Lengkap dengan Pandas ==========
print("\n" + "=" * 50)
print("CONTOH 3: Analisis Lengkap dengan Pandas")
print("=" * 50)

# Buat dataset
data = {
    'Bulan': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'Penjualan_Aktual': [120, 135, 150, 145, 160, 175, 170, 185, 190, 200],
    'Penjualan_Prediksi': [118, 140, 148, 150, 158, 178, 168, 188, 192, 198]
}

df = pd.DataFrame(data)
df['Error'] = df['Penjualan_Aktual'] - df['Penjualan_Prediksi']
df['Squared_Error'] = df['Error'] ** 2

print(df)

# Hitung metrik
rmse = np.sqrt(df['Squared_Error'].mean())
mae = df['Error'].abs().mean()
std_error = df['Error'].std()

print(f"\nMetrik Evaluasi:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"Standard Deviation dari Error: {std_error:.4f}")

# Statistik deskriptif
print(f"\nStatistik Penjualan Aktual:")
print(f"Mean: {df['Penjualan_Aktual'].mean():.2f}")
print(f"Std Dev: {df['Penjualan_Aktual'].std():.2f}")
print(f"Min: {df['Penjualan_Aktual'].min()}")
print(f"Max: {df['Penjualan_Aktual'].max()}")

# ========== CONTOH 4: Visualisasi ==========
print("\n" + "=" * 50)
print("CONTOH 4: Membuat Visualisasi")
print("=" * 50)

plt.figure(figsize=(12, 5))
# ========== CONTOH 4: Visualisasi ==========
print("\n" + "=" * 50)
print("CONTOH 4: Membuat Visualisasi")
print("=" * 50)

plt.figure(figsize=(12, 5))

# Plot 1: Perbandingan Aktual vs Prediksi
plt.subplot(1, 2, 1)
plt.plot(df['Bulan'], df['Penjualan_Aktual'], marker='o', label='Aktual', linewidth=2)
plt.plot(df['Bulan'], df['Penjualan_Prediksi'], marker='s', label='Prediksi', linewidth=2)
plt.xlabel('Bulan')
plt.ylabel('Penjualan')
plt.title('Perbandingan Penjualan Aktual vs Prediksi')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Distribusi Error
plt.subplot(1, 2, 2)
plt.bar(df['Bulan'], df['Error'], color=['red' if x < 0 else 'green' for x in df['Error']])
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Bulan')
plt.ylabel('Error')
plt.title('Distribusi Error Prediksi')
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Simpan gambar
plt.savefig('analisis_rmse_sd.png', dpi=300, bbox_inches='tight')
print("Visualisasi disimpan sebagai 'analisis_rmse_sd.png'")

# TAMBAHKAN BARIS INI AGAR VISUALISASI MUNCUL
plt.show()

# Plot 1: Perbandingan Aktual vs Prediksi
plt.subplot(1, 2, 1)
plt.plot(df['Bulan'], df['Penjualan_Aktual'], marker='o', label='Aktual', linewidth=2)
plt.plot(df['Bulan'], df['Penjualan_Prediksi'], marker='s', label='Prediksi', linewidth=2)
plt.xlabel('Bulan')
plt.ylabel('Penjualan')
plt.title('Perbandingan Penjualan Aktual vs Prediksi')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Distribusi Error
plt.subplot(1, 2, 2)
plt.bar(df['Bulan'], df['Error'], color=['red' if x < 0 else 'green' for x in df['Error']])
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Bulan')
plt.ylabel('Error')
plt.title('Distribusi Error Prediksi')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_rmse_sd.png', dpi=300, bbox_inches='tight')
print("Visualisasi disimpan sebagai 'analisis_rmse_sd.png'")

# ========== CONTOH 5: Fungsi Reusable ==========
print("\n" + "=" * 50)
print("CONTOH 5: Fungsi yang Bisa Digunakan Ulang")
print("=" * 50)

def hitung_metrik(actual, predicted):
    """
    Menghitung berbagai metrik evaluasi
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Error
    errors = actual - predicted
    
    # RMSE
    rmse = np.sqrt(np.mean(errors**2))
    
    # MAE
    mae = np.mean(np.abs(errors))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs(errors / actual)) * 100
    
    # R-squared
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Standard Deviation
    std_actual = np.std(actual, ddof=1)
    std_predicted = np.std(predicted, ddof=1)
    std_errors = np.std(errors, ddof=1)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R-squared': r_squared,
        'Std_Actual': std_actual,
        'Std_Predicted': std_predicted,
        'Std_Errors': std_errors
    }

# Test fungsi
hasil = hitung_metrik(df['Penjualan_Aktual'], df['Penjualan_Prediksi'])

print("Hasil Perhitungan Metrik:")
for key, value in hasil.items():
    print(f"{key}: {value:.4f}")

print("\n" + "=" * 50)
print("SELESAI")
print("=" * 50)
# ========== CONTOH 6: Analisis Korelasi ==========
print("\n" + "=" * 50)
print("CONTOH 6: Menghitung Korelasi")
print("=" * 50)

# --- 6A: Korelasi Manual (Pearson) ---
print("\n--- 6A: Korelasi Manual ---")

x = np.array(df['Penjualan_Aktual'])
y = np.array(df['Penjualan_Prediksi'])

mean_x = np.mean(x)
mean_y = np.mean(y)

pembilang = np.sum((x - mean_x) * (y - mean_y))
penyebut = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))

korelasi_manual = pembilang / penyebut
print(f"Korelasi Pearson (Manual): {korelasi_manual:.4f}")

# --- 6B: Korelasi dengan NumPy ---
print("\n--- 6B: Korelasi dengan NumPy ---")

korelasi_numpy = np.corrcoef(x, y)[0, 1]
print(f"Korelasi Pearson (NumPy): {korelasi_numpy:.4f}")

# --- 6C: Korelasi dengan Pandas ---
print("\n--- 6C: Korelasi dengan Pandas ---")

korelasi_pandas = df['Penjualan_Aktual'].corr(df['Penjualan_Prediksi'])
print(f"Korelasi Pearson (Pandas): {korelasi_pandas:.4f}")

# --- 6D: Interpretasi Nilai Korelasi ---
print("\n--- 6D: Interpretasi Korelasi ---")

def interpretasi_korelasi(r):
    """Memberikan interpretasi dari nilai korelasi"""
    r_abs = abs(r)
    if r_abs == 1.0:
        kekuatan = "Sempurna"
    elif r_abs >= 0.80:
        kekuatan = "Sangat Kuat"
    elif r_abs >= 0.60:
        kekuatan = "Kuat"
    elif r_abs >= 0.40:
        kekuatan = "Sedang"
    elif r_abs >= 0.20:
        kekuatan = "Lemah"
    else:
        kekuatan = "Sangat Lemah"

    arah = "Positif" if r > 0 else "Negatif"
    return kekuatan, arah

kekuatan, arah = interpretasi_korelasi(korelasi_pandas)
print(f"Nilai Korelasi : {korelasi_pandas:.4f}")
print(f"Arah           : {arah}")
print(f"Kekuatan       : {kekuatan}")

# --- 6E: Matriks Korelasi dengan Dataset Lebih Luas ---
print("\n--- 6E: Matriks Korelasi ---")

# Tambah kolom baru untuk memperkaya analisis
df['Biaya_Iklan']   = [20, 22, 25, 23, 28, 30, 29, 32, 35, 38]
df['Jumlah_Pembeli'] = [40, 45, 50, 48, 55, 60, 58, 63, 67, 72]

# Hitung matriks korelasi
kolom_analisis = ['Penjualan_Aktual', 'Penjualan_Prediksi', 'Biaya_Iklan', 'Jumlah_Pembeli']
matriks_korelasi = df[kolom_analisis].corr()

print("Matriks Korelasi:")
print(matriks_korelasi.round(4))

# --- 6F: Korelasi Spearman (untuk data tidak normal) ---
print("\n--- 6F: Korelasi Spearman ---")

korelasi_spearman = df['Penjualan_Aktual'].corr(df['Penjualan_Prediksi'], method='spearman')
print(f"Korelasi Spearman: {korelasi_spearman:.4f}")
print("(Spearman digunakan jika data tidak berdistribusi normal)")

# ========== CONTOH 7: Visualisasi Korelasi ==========
print("\n" + "=" * 50)
print("CONTOH 7: Visualisasi Korelasi")
print("=" * 50)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: Scatter Plot Aktual vs Prediksi ---
axes[0].scatter(df['Penjualan_Aktual'], df['Penjualan_Prediksi'],
                color='blue', alpha=0.7, s=100, edgecolors='black')

# Garis regresi
m, b = np.polyfit(df['Penjualan_Aktual'], df['Penjualan_Prediksi'], 1)
x_line = np.linspace(df['Penjualan_Aktual'].min(), df['Penjualan_Aktual'].max(), 100)
axes[0].plot(x_line, m * x_line + b, color='red', linewidth=2, label=f'y = {m:.2f}x + {b:.2f}')

axes[0].set_xlabel('Penjualan Aktual')
axes[0].set_ylabel('Penjualan Prediksi')
axes[0].set_title(f'Scatter Plot\nKorelasi = {korelasi_pandas:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Plot 2: Heatmap Matriks Korelasi ---
im = axes[1].imshow(matriks_korelasi, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=axes[1])

ticks = range(len(kolom_analisis))
axes[1].set_xticks(ticks)
axes[1].set_yticks(ticks)
label_pendek = ['Aktual', 'Prediksi', 'Iklan', 'Pembeli']
axes[1].set_xticklabels(label_pendek, rotation=30, ha='right')
axes[1].set_yticklabels(label_pendek)
axes[1].set_title('Heatmap Matriks Korelasi')

# Tambahkan angka di tiap sel heatmap
for i in range(len(kolom_analisis)):
    for j in range(len(kolom_analisis)):
        axes[1].text(j, i, f"{matriks_korelasi.iloc[i, j]:.2f}",
                     ha='center', va='center', fontsize=10,
                     color='white' if abs(matriks_korelasi.iloc[i, j]) > 0.5 else 'black')

# --- Plot 3: Bar Chart Korelasi terhadap Penjualan Aktual ---
korelasi_vs_aktual = df[kolom_analisis].corr()['Penjualan_Aktual'].drop('Penjualan_Aktual')
warna = ['green' if v > 0 else 'red' for v in korelasi_vs_aktual]

axes[2].bar(korelasi_vs_aktual.index, korelasi_vs_aktual.values, color=warna, edgecolor='black')
axes[2].axhline(y=0, color='black', linewidth=0.8)
axes[2].set_ylim(-1.1, 1.1)
axes[2].set_ylabel('Nilai Korelasi')
axes[2].set_title('Korelasi terhadap Penjualan Aktual')
axes[2].set_xticklabels(korelasi_vs_aktual.index, rotation=15, ha='right')
axes[2].grid(True, alpha=0.3)

# Tambahkan label nilai di tiap bar
for i, (idx, val) in enumerate(korelasi_vs_aktual.items()):
    axes[2].text(i, val + 0.02 if val >= 0 else val - 0.05,
                 f"{val:.2f}", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('analisis_korelasi.png', dpi=300, bbox_inches='tight')
print("Visualisasi disimpan sebagai 'analisis_korelasi.png'")
plt.show()

# ========== CONTOH 8: Fungsi Korelasi Reusable ==========
print("\n" + "=" * 50)
print("CONTOH 8: Fungsi Korelasi yang Bisa Digunakan Ulang")
print("=" * 50)

def hitung_korelasi(data, kolom_target):
    """
    Menghitung korelasi semua kolom numerik terhadap kolom target.

    Parameter:
        data          : DataFrame pandas
        kolom_target  : nama kolom yang jadi acuan (string)

    Return:
        DataFrame berisi nilai korelasi + interpretasinya
    """
    hasil = []
    kolom_numerik = data.select_dtypes(include=[np.number]).columns

    for kolom in kolom_numerik:
        if kolom == kolom_target:
            continue

        r_pearson  = data[kolom_target].corr(data[kolom], method='pearson')
        r_spearman = data[kolom_target].corr(data[kolom], method='spearman')
        kekuatan, arah = interpretasi_korelasi(r_pearson)

        hasil.append({
            'Kolom'             : kolom,
            'Pearson'           : round(r_pearson, 4),
            'Spearman'          : round(r_spearman, 4),
            'Kekuatan'          : kekuatan,
            'Arah'              : arah
        })

    return pd.DataFrame(hasil).sort_values('Pearson', ascending=False)


# Jalankan fungsi
hasil_korelasi = hitung_korelasi(df, 'Penjualan_Aktual')
print(f"\nKorelasi semua variabel terhadap 'Penjualan_Aktual':")
print(hasil_korelasi.to_string(index=False))

print("\n" + "=" * 50)
print("SELESAI")
print("=" * 50)
# ========== CONTOH 9: Regresi Linear Sederhana ==========
print("\n" + "=" * 50)
print("CONTOH 9: Regresi Linear Sederhana")
print("=" * 50)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- 9A: Regresi Linear Manual ---
print("\n--- 9A: Regresi Linear Manual ---")

x = np.array(df['Penjualan_Aktual']).reshape(-1, 1)
y = np.array(df['Penjualan_Prediksi'])

# Hitung koefisien regresi (slope & intercept) secara manual
n        = len(x)
mean_x   = np.mean(x)
mean_y   = np.mean(y)
slope    = np.sum((x.flatten() - mean_x) * (y - mean_y)) / np.sum((x.flatten() - mean_x)**2)
intercept = mean_y - slope * mean_x

print(f"Persamaan Regresi : y = {slope:.4f}x + {intercept:.4f}")
print(f"Slope (kemiringan) : {slope:.4f}")
print(f"Intercept (titik potong) : {intercept:.4f}")

# Prediksi dengan persamaan regresi manual
y_pred_manual = slope * x.flatten() + intercept
r2_manual = 1 - np.sum((y - y_pred_manual)**2) / np.sum((y - mean_y)**2)
print(f"R² (Manual)       : {r2_manual:.4f}")

# --- 9B: Regresi Linear dengan Sklearn ---
print("\n--- 9B: Regresi Linear dengan Sklearn ---")

model_linear = LinearRegression()
model_linear.fit(x, y)

print(f"Slope (sklearn)    : {model_linear.coef_[0]:.4f}")
print(f"Intercept (sklearn): {model_linear.intercept_:.4f}")

y_pred_linear = model_linear.predict(x)
r2_linear = r2_score(y, y_pred_linear)
rmse_linear = np.sqrt(np.mean((y - y_pred_linear)**2))
print(f"R² (sklearn)       : {r2_linear:.4f}")
print(f"RMSE               : {rmse_linear:.4f}")

# ========== CONTOH 10: Regresi Polinomial ==========
print("\n" + "=" * 50)
print("CONTOH 10: Regresi Polinomial (Derajat 2 & 3)")
print("=" * 50)

hasil_poly = {}

for derajat in [2, 3]:
    model_poly = make_pipeline(PolynomialFeatures(derajat), LinearRegression())
    model_poly.fit(x, y)

    y_pred_poly = model_poly.predict(x)
    r2_poly     = r2_score(y, y_pred_poly)
    rmse_poly   = np.sqrt(np.mean((y - y_pred_poly)**2))

    hasil_poly[derajat] = {
        'model'  : model_poly,
        'y_pred' : y_pred_poly,
        'r2'     : r2_poly,
        'rmse'   : rmse_poly
    }

    print(f"\nPolinomial Derajat {derajat}:")
    print(f"  R²   : {r2_poly:.4f}")
    print(f"  RMSE : {rmse_poly:.4f}")

# ========== CONTOH 11: Regresi Multi Variabel ==========
print("\n" + "=" * 50)
print("CONTOH 11: Regresi Multi Variabel")
print("=" * 50)

# Gunakan Biaya_Iklan & Jumlah_Pembeli sebagai prediktor
X_multi = df[['Biaya_Iklan', 'Jumlah_Pembeli']].values
y_multi = df['Penjualan_Aktual'].values

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

y_pred_multi = model_multi.predict(X_multi)
r2_multi     = r2_score(y_multi, y_pred_multi)
rmse_multi   = np.sqrt(np.mean((y_multi - y_pred_multi)**2))

print(f"Koefisien Biaya_Iklan   : {model_multi.coef_[0]:.4f}")
print(f"Koefisien Jumlah_Pembeli: {model_multi.coef_[1]:.4f}")
print(f"Intercept               : {model_multi.intercept_:.4f}")
print(f"R²                      : {r2_multi:.4f}")
print(f"RMSE                    : {rmse_multi:.4f}")
print(f"\nPersamaan: Penjualan = {model_multi.coef_[0]:.2f}*Iklan + {model_multi.coef_[1]:.2f}*Pembeli + {model_multi.intercept_:.2f}")

# ========== CONTOH 12: Visualisasi Kurva Regresi ==========
print("\n" + "=" * 50)
print("CONTOH 12: Visualisasi Kurva Regresi")
print("=" * 50)

# Buat x halus untuk kurva mulus
x_halus = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analisis Kurva Regresi', fontsize=16, fontweight='bold')

# --- Plot 1: Regresi Linear ---
ax1 = axes[0, 0]
ax1.scatter(x, y, color='blue', s=80, zorder=5, label='Data Asli', edgecolors='black')
ax1.plot(x_halus, model_linear.predict(x_halus),
         color='red', linewidth=2, label=f'Linear (R²={r2_linear:.4f})')
ax1.set_xlabel('Penjualan Aktual')
ax1.set_ylabel('Penjualan Prediksi')
ax1.set_title('Regresi Linear')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Plot 2: Regresi Polinomial Derajat 2 ---
ax2 = axes[0, 1]
ax2.scatter(x, y, color='blue', s=80, zorder=5, label='Data Asli', edgecolors='black')
ax2.plot(x_halus, hasil_poly[2]['model'].predict(x_halus),
         color='green', linewidth=2, label=f'Polinom D2 (R²={hasil_poly[2]["r2"]:.4f})')
ax2.set_xlabel('Penjualan Aktual')
ax2.set_ylabel('Penjualan Prediksi')
ax2.set_title('Regresi Polinomial Derajat 2')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Plot 3: Regresi Polinomial Derajat 3 ---
ax3 = axes[1, 0]
ax3.scatter(x, y, color='blue', s=80, zorder=5, label='Data Asli', edgecolors='black')
ax3.plot(x_halus, hasil_poly[3]['model'].predict(x_halus),
         color='purple', linewidth=2, label=f'Polinom D3 (R²={hasil_poly[3]["r2"]:.4f})')
ax3.set_xlabel('Penjualan Aktual')
ax3.set_ylabel('Penjualan Prediksi')
ax3.set_title('Regresi Polinomial Derajat 3')
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Plot 4: Perbandingan Semua Kurva ---
ax4 = axes[1, 1]
ax4.scatter(x, y, color='blue', s=80, zorder=5, label='Data Asli', edgecolors='black')
ax4.plot(x_halus, model_linear.predict(x_halus),
         color='red',    linewidth=2, label=f'Linear      (R²={r2_linear:.4f})')
ax4.plot(x_halus, hasil_poly[2]['model'].predict(x_halus),
         color='green',  linewidth=2, label=f'Polinom D2 (R²={hasil_poly[2]["r2"]:.4f})', linestyle='--')
ax4.plot(x_halus, hasil_poly[3]['model'].predict(x_halus),
         color='purple', linewidth=2, label=f'Polinom D3 (R²={hasil_poly[3]["r2"]:.4f})', linestyle=':')
ax4.set_xlabel('Penjualan Aktual')
ax4.set_ylabel('Penjualan Prediksi')
ax4.set_title('Perbandingan Semua Kurva Regresi')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kurva_regresi.png', dpi=300, bbox_inches='tight')
print("Visualisasi disimpan sebagai 'kurva_regresi.png'")
plt.show()

# ========== CONTOH 13: Tabel Ringkasan Model ==========
print("\n" + "=" * 50)
print("CONTOH 13: Ringkasan Perbandingan Model")
print("=" * 50)

ringkasan = pd.DataFrame({
    'Model'  : ['Linear', 'Polinomial D2', 'Polinomial D3', 'Multi Variabel'],
    'R²'     : [r2_linear,
                hasil_poly[2]['r2'],
                hasil_poly[3]['r2'],
                r2_multi],
    'RMSE'   : [rmse_linear,
                hasil_poly[2]['rmse'],
                hasil_poly[3]['rmse'],
                rmse_multi],
    'Keterangan': [
        'y = ax + b',
        'y = ax² + bx + c',
        'y = ax³ + bx² + cx + d',
        'y = a*Iklan + b*Pembeli + c'
    ]
})

ringkasan['R²']   = ringkasan['R²'].round(4)
ringkasan['RMSE'] = ringkasan['RMSE'].round(4)

print(ringkasan.to_string(index=False))

# Tentukan model terbaik
model_terbaik = ringkasan.loc[ringkasan['R²'].idxmax(), 'Model']
print(f"\n✅ Model terbaik: {model_terbaik} (R² tertinggi = {ringkasan['R²'].max():.4f})")

print("\n" + "=" * 50)
print("SELESAI")
print("=" * 50)
