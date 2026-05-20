# 📊 Analisis Data: RMSE, Standard Deviation & Regresi Linear

> Tugas Mata Kuliah Analisis Data  
> Implementasi Python untuk menghitung dan memvisualisasikan metrik evaluasi model machine learning

---

## 📌 Deskripsi

Proyek ini membahas konsep dasar analisis data menggunakan Python, mulai dari menghitung error model (RMSE), mengukur sebaran data (Standard Deviation), menganalisis hubungan antar variabel (Korelasi), hingga membangun model prediksi (Regresi Linear).

Seluruh kode ditulis dengan **3 pendekatan berbeda** (manual, NumPy, dan sklearn) agar logika di balik setiap rumus benar-benar dipahami, bukan sekadar dijalankan.

---

## 🗂️ Struktur File

```
📁 analisis-data-rmse-sd/
├── 📄 Model_Analisis_Data2_RMSE_SD.py         # File asli (versi lengkap)
├── 📄 Model_Analisis_Data2_RMSE_SD_Colab.py   # Versi Google Colab (3 sel)
└── 📄 README.md                                # Dokumentasi ini
```

---

## 📚 Isi Materi (10 Bagian)

### 🔵 SEL 1 — RMSE & Standard Deviation

| Bagian | Topik | Penjelasan |
|--------|-------|------------|
| 1 | **RMSE** | Mengukur rata-rata selisih antara nilai aktual dan prediksi |
| 2 | **Standard Deviation** | Mengukur seberapa menyebar data dari nilai rata-ratanya |
| 3 | **Analisis Penjualan** | Tabel error per bulan + visualisasi grafik |

### 🟡 SEL 2 — Analisis Korelasi

| Bagian | Topik | Penjelasan |
|--------|-------|------------|
| 4 | **Korelasi Pearson** | Mengukur kekuatan hubungan linear antar dua variabel |
| 5 | **Visualisasi Korelasi** | Scatter plot, heatmap matriks, dan bar chart korelasi |

### 🟢 SEL 3 — Regresi Linear

| Bagian | Topik | Penjelasan |
|--------|-------|------------|
| 6 | **Regresi Linear Sederhana** | Mencari garis terbaik y = ax + b |
| 7 | **Regresi Polinomial** | Kurva derajat 2 dan 3 |
| 8 | **Regresi Multi Variabel** | Dua prediktor sekaligus (Iklan + Pembeli) |
| 9 | **Visualisasi Kurva Regresi** | Perbandingan 4 model dalam satu tampilan |
| 10 | **Ringkasan Model** | Tabel perbandingan R² dan RMSE semua model |

---

## 📐 Rumus yang Digunakan

```
RMSE  = √( (1/n) × Σ (aktual - prediksi)² )

SD    = √( (1/N) × Σ (xi - mean)² )          ← populasi
SD    = √( (1/N-1) × Σ (xi - mean)² )        ← sampel

r     = Σ(xi - x̄)(yi - ȳ) / √[Σ(xi-x̄)² × Σ(yi-ȳ)²]

R²    = 1 - (SS_residual / SS_total)
```

---

## 📊 Hasil Eksekusi

### RMSE — Contoh Data Aktual vs Prediksi
```
RMSE (Manual)  : 3.2094
RMSE (NumPy)   : 3.2094
RMSE (sklearn) : 3.2094
→ Ketiga cara menghasilkan nilai yang sama ✔
```

### Standard Deviation — Data Nilai Siswa (n=15)
```
Mean            : 83.93
SD Populasi     : 7.28
SD Sampel       : 7.54
```

### Korelasi — Penjualan Aktual vs Prediksi
```
Pearson   : 0.9928  → Sangat Kuat, Positif
Spearman  : 0.9879
```

### Perbandingan Model Regresi
```
Model            R²      RMSE    Persamaan
Linear           0.9856  2.9201  y = ax + b
Polinomial D2    0.9860  2.8740  y = ax² + bx + c
Polinomial D3    0.9862  2.8547  y = ax³ + bx² + cx + d
Multi Variabel   0.9970  1.3211  y = a·Iklan + b·Pembeli + c

✅ Model terbaik: Multi Variabel (R² = 0.9970 → menjelaskan 99.7% variasi data)
```

---

## 🛠️ Teknologi yang Digunakan

| Library | Fungsi |
|---------|--------|
| `numpy` | Operasi array dan perhitungan numerik |
| `pandas` | Pengelolaan data dalam bentuk tabel (DataFrame) |
| `scikit-learn` | Model regresi dan metrik evaluasi |
| `matplotlib` | Visualisasi grafik dan plot |

---

## ▶️ Cara Menjalankan

### Di VS Code
```bash
# 1. Pastikan library sudah terinstall
pip install numpy pandas scikit-learn matplotlib

# 2. Jalankan file
python Model_Analisis_Data2_RMSE_SD_Colab.py
```

### Di Google Colab
1. Buka [colab.research.google.com](https://colab.research.google.com)
2. Buat notebook baru → **File > New Notebook**
3. Salin isi file `_Colab.py` ke **3 sel terpisah** sesuai tanda `# ══ SEL 1/2/3`
4. Jalankan satu per satu dengan **Shift+Enter**

---

## 🎓 Kesimpulan Pembelajaran

1. **RMSE** yang kecil menunjukkan model prediksi yang lebih akurat
2. **Standard Deviation** yang besar berarti data sangat bervariasi
3. **Korelasi 0.99** berarti hubungan antara aktual dan prediksi sangat erat
4. **R² mendekati 1** berarti model mampu menjelaskan hampir seluruh pola data
5. Model **Multi Variabel** terbukti lebih baik karena menggunakan lebih banyak informasi

---

## 👤 Informasi

**Mata Kuliah** : Analisis Data  
**Bahasa**      : Python 3  
**Platform**    : VS Code / Google Colab  

---

> *"Memahami data bukan hanya soal menjalankan kode, tapi memahami logika di balik setiap angka."*
