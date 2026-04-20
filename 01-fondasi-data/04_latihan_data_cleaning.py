"""
=============================================================
FASE 1 — MODUL 4: DATA CLEANING DENGAN PANDAS
=============================================================
Di dunia nyata, data TIDAK PERNAH bersih. Masalah umum:
1. Missing values (NaN, kosong, "N/A", "-")
2. Duplikat data
3. Tipe data salah (angka tersimpan sebagai string)
4. Format tidak konsisten ("Jakarta", "JAKARTA", "jakarta")
5. Outlier / nilai tidak masuk akal
6. Kolom tidak relevan

Pandas CUKUP untuk semua ini. File ini akan simulasi dataset
"kotor" yang realistis dan membersihkannya step by step.

Durasi target: 2-3 jam
=============================================================
"""

import numpy as np
import pandas as pd

# ===========================================================
# 📖 BAGIAN 1: Buat Dataset "Kotor" yang Realistis
# ===========================================================
# Simulasi data penjualan toko online — seperti export dari database
# yang belum pernah di-validasi. Ini mirip kondisi data di dunia nyata.

np.random.seed(42)
n = 200
"""
Docstring untuk inisialisasi dataset:
Fungsi ini menginisialisasi parameter dasar untuk membuat dataset simulasi:
- np.random.seed(42): Menetapkan seed untuk random number generator agar hasil 
    reproducible (konsisten setiap kali dijalankan). Angka 42 adalah seed yang umum 
    digunakan sebagai standar dalam machine learning.
- n = 200: Mendefinisikan jumlah baris/record data yang akan dibuat, yaitu 200 
    data penjualan simulasi untuk keperluan praktek data cleaning.
Kedua baris ini memastikan bahwa data yang dibuat bersifat deterministic dan 
dapat diulang dengan hasil yang sama, penting untuk pembelajaran dan debugging.
"""

# Data "kotor" - perhatikan semua masalah yang sengaja ditanam
raw_data = pd.DataFrame({
    # ID transaksi: ada duplikat (transaksi ke-3 tercatat 2x)
    'transaction_id': [f"TRX-{i:04d}" for i in range(1, n+1)],

    # Tanggal: format CAMPUR ADUK (dd/mm/yyyy, yyyy-mm-dd, dd-Mon-yyyy)
    'date': np.random.choice([
        '15/01/2024', '2024-01-16', '17-Jan-2024', '18/01/2024',
        '2024-01-19', '20-Jan-2024', '21/01/2024', '2024-01-22',
        '23-Jan-2024', '24/01/2024', None,  # ada yang kosong
    ], size=n),

    # Nama customer: inkonsisten (huruf besar/kecil, spasi ekstra)
    'customer_name': np.random.choice([
        'Budi Santoso', 'budi santoso', 'BUDI SANTOSO',  # orang sama, 3 format
        'Siti Aminah', 'siti aminah ', ' Siti Aminah',   # spasi ekstra
        'Ahmad Rizki', 'Dewi Lestari', 'Eko Prasetyo',
        'Fitri Handayani', None, 'N/A', '-',             # missing dengan berbagai format
    ], size=n),

    # Kota: inkonsisten
    'city': np.random.choice([
        'Jakarta', 'jakarta', 'JAKARTA', 'Jkt',          # kota sama, 4 format
        'Surabaya', 'surabaya', 'SBY',                   # kota sama, 3 format
        'Bandung', 'bandung', 'BDG',                     # kota sama, 3 format
        'Yogyakarta', 'Jogja', None,
    ], size=n),

    # Harga: ada yang string (dari copy-paste Excel), ada negatif, ada 0
    'price': np.random.choice([
        50000, 75000, 100000, 150000, 250000, 500000,
        '75.000', '100.000', 'Rp 250000',               # string, bukan angka!
        -50000, 0, None, 999999999],                     # negatif, nol, kosong, outlier
    size=n),

    # Quantity: ada float yang harusnya integer, ada negatif
    'quantity': np.random.choice([
        1, 2, 3, 5, 10, 1.0, 2.0,
         -1, 0, None, 1000 # negatif, nol, kosong, outlier
    ], size=n),

    # Kategori produk: ada typo dan inkonsisten
    'category': np.random.choice([
        'Elektronik', 'elektronik', 'ELEKTRONIK', 'Eletronik',  # typo: Eletronik
        'Fashion', 'fashion', 'Fshion',                          # typo: Fshion
        'Makanan', 'makanan', 'Makan',                           # salah: Makan
        'Kesehatan', None,
    ], size=n),

    # Rating: 1-5, tapi ada yang di luar range dan string
    'rating': np.random.choice([
        1,2,3,4,5,0, -1, 6, 10, None, 'bagus', '4'
    ], size=n),
})

# Tambahkan beberapa baris duplikat (simulasi data tercatat 2x)
duplicates = raw_data.iloc[2:5].copy()
raw_data = pd.concat([raw_data, duplicates], ignore_index=True)

print("=" * 60)
print("DATASET MENTAH (KOTOR)")
print("=" * 60)
print(f"Shape: {raw_data.shape}")
print(f"\nInfo: \n")
print(raw_data.info())
print(f"\nSample 5 baris pertama:")
print(raw_data.head())
print(f"\nSample 5 baris terakhir (termasuk duplikat):")
print(raw_data.tail())

# ===========================================================
# 📖 BAGIAN 2: Inspeksi — Kenali Masalahnya Dulu
# ===========================================================
# ATURAN #1 DATA CLEANING: JANGAN langsung bersihkan.
# Pahami dulu data kamu. Lihat pola masalahnya.

print("\n"+"=" * 60)
print("STEP 1: INSPEKSI DATA")
print("=" * 60)

# 2a. Cek missing values
# isnull().sum() menghitung jumlah Nan per kolom
# tapi ingat: missing tidak selalu NaN! Bisa juga "N/A", "-", "", 0
print("\n--- Missing Values (NaN saja) ---")
print(raw_data.isnull().sum())

#2b. Cek "hidden" missing values - yang bukan NaN tapi sebenarnya kosong
print("\n--- Hidden Missing: 'N/A', '-', '' di customer_name ---")
hidden_missing = raw_data['customer_name'].isin(['N/A', '-', '', ' '])
print(f"Jumlah: {hidden_missing.sum()}")

#2c. Cek tipe data - sering jadi sumber bug
# kolom price dan quantity mungkin jadi object (string) karena ada campuran
print("\n--- Tipe Data ---")
print(raw_data.dtypes)

#2d. Cek duplikat
print(f"\n--- Duplikat ---")
print(f"Jumlah baris duplikat: {raw_data.duplicated().sum()}")

#2e. Cek nilai unik untuk kolom kategorikal (lihat inkonsistensi)
print("\n---Nilai Unik: city ---")
print(raw_data['city'].value_counts())
print("\n--- Nilai Unik: category ---")
print(raw_data["category"].value_counts())

# ===========================================================
# 📖 BAGIAN 3: Cleaning Step-by-Step
# ===========================================================
# Selalu buat COPY sebelum cleaning. Jangan ubah data asli.
# Kalau ada yang salah, kita masih bisa cek data mentahnya.

df = raw_data.copy()

print("\n" + "=" * 60)
print("STEP 2: MULAI CLEANING")
print("=" * 60)

# ------ STEP 2a: Hapus Duplikat ------
# drop_duplicates() menghapus baris yang identik semua kolomnya
# keep='first' = simpan kemunculan pertama, hapus sisanya
before = len(df)
df = df.drop_duplicates(keep='first')
after = len(df)
print(f"\n[2a] Hapus duplikat: {before} → {after} baris ({before - after} dihapus)")

# ------ STEP 2b: Standardisasi Missing Values ------
# Ganti semua variasi "kosong" menjadi NaN yang konsisten
# Supaya nanti bisa ditangani seragam dengan fillna() atau dropna()
missing_indicators = ['N/A', 'n/a', 'NA', '-', '', ' ', 'null', 'NULL', 'None']
df = df.replace(missing_indicators, np.nan)

print(f"\n[2b] Setelah standardisasi missing values:")
print(df.isnull().sum())

# ----- STEP 2c: Bersihkan & Konversi Kolom 'price' ------
# Masalah: ada "75.000", "Rp 250000", angka negatif, 0, outlier
# Strategi:
# 1. Hilangkan karakter non-numerik (Rp, titik pemisah ribuan)
# 2. Konversi ke float
# 3. Hapus/ganti nilai tidak valid (negatif, 0, outlier)

def clean_price(val):
    """Bersihkan nilai harga dari format yang tidak konsisten"""
    if pd.isna(val):
        return np.nan
    val = str(val)                  # pastikan string
    val = val.replace("Rp", "")     # hapus "Rp"
    val = val.replace(".", "")      # hapus titik pemisah ribuan
    val = val.replace(",", "")      # hapus koma jika ada
    val = val.strip()               # hapus spasi
    try:
        return float(val)
    except ValueError:
        return np.nan               # kalau tetap gagal → NaN
    

df["price"] = df["price"].apply(clean_price)

# Tandai harga tidak valid: negatif atau 0
df.loc[df["price"] <= 0, "price"] = np.nan

# Tangani outlier harga: gunakan IQR method
# IQR = Interquartile Range = Q3 - Q1
# Nilai di luar [Q1- 1.5*IQR, Q3 + 1.5*IQR] dianggap outlier
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_price = (df['price'] < lower_bound) | (df['price'] > upper_bound) # yang di bawah atau di atas IQR
print(f"\n[2c] Outlier harga (IQR method): {outliers_price.sum()} baris")
print(f"     Range valid: {lower_bound:,.0f} - {upper_bound:,.0f}")

# Ganti outlier dengan NaN (nanti akan dihandle di step imputasi)
df.loc[outliers_price, 'price'] = np.nan

print(f"    Price setelah cleaning: {df['price'].describe()}")

# ------ STEP 2d: Bersihkan & Konversi Kolom 'quantity' ------
# Konversi ke numeric, paksa error jadi NaN
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

# Quantity harus integer positif
df.loc[df['quantity'] <= 0, 'quantity'] = np.nan

# Outlier quantity (>100 tidak masuk akal untuk 1 transaksi retail)
df.loc[df['quantity'] > 100, 'quantity'] = np.nan

# Konversi ke integer (Int64 mendukung NaN, int biasa tidak)
df['quantity'] = df['quantity'].astype('Int64')

print(f"\n[2d] Quantity setelah cleaning: {df['quantity'].describe()}")

# ------ STEP 2e: Standardisasi Nama Customer ------
# Masalah: "Budi Santoso", "budi santoso", "BUDI SANTOSO" = orang yang sama
# Strategi: strip spasi + title case

df['customer_name'] = (df['customer_name']
                       .str.strip()             # hapus spasi di awal/akhir
                       .str.title())            # "budi santoso" → "Budi Santoso"

print(f"\n[2e] Customer untuk setelah cleaning: {df['customer_name'].nunique()}")
print(df['customer_name'].value_counts().head(10))

# ------ STEP 2f: Standardisasi Kolom 'city' ------
# Masalah: "Jakarta", "jakarta", "JAKARTA", "Jkt" = kota yang sama
# Strategi: lowercase dulu, lalu mapping singkatan → nama lengkap

df['city'] = df['city'].str.strip().str.lower()
print(df['city'].value_counts())
# Mapping singkatan dan variasi ke nama standard
city_mapping = {
    'jakarta': 'Jakarta',
    'jkt': 'Jakarta',
    'surabaya': 'Surabaya',
    'sby': 'Surabaya',
    'bdg': 'Bandung',
    'bandung': 'Bandung',
    'yogyakarta': 'Yogyakarta',
    'jogja': 'Yogyakarta',
}

df['city'] = df['city'].map(city_mapping)
# catatan: city yang tidak ada di mapping akan jadi NaN (karena map, bukan replace)

print(f"\n[2f] City setelah cleaning:")
print(df['city'].value_counts(dropna=False))

# ------ STEP 2g: Standardisasi Kolom 'category' ------
# Masalah: inkonsistensi haruf besar/kecil + typo
# Strategi: lowercase → mapping koreksi typo

df['category'] = df['category'].str.strip().str.lower()
print(df['category'].value_counts())

category_mapping = {
    'elektronik': 'Elektronik',
    'eletronik': 'Elektronik',
    'fashion': 'Fashion',
    'fshion': 'Fashion',
    'makan': 'Makanan',
    'makanan': 'Makanan',
    'kesehatan': 'Kesehatan',
}
df['category'] = df['category'].map(category_mapping)

print(f"\n[2g] Category setelah cleaning: ")
print(df['category'].value_counts(dropna=False))

# ------ STEP 2h: Bersihkan Kolom 'rating' ------
# Konversi ke numeric (string "bagus" → NaN, string "4" → 4.0)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# rating valid: 1-5 saja
df.loc[(df['rating'] < 1) | (df['rating'] > 5), 'rating'] = np.nan
print(df['rating'].value_counts())
print(f"\n[2h] Rating setelah cleaning: {df['rating'].describe()}")

# ------ STEP 2i: Bersihkan Kolom 'date' ------
# Format campur aduk: 'dd/mm/yyyy', 'yyyy-mm-dd', 'dd-Mon-yyyy'
# pd.to_datetime dengan dayfirst=True dan format campuran
# infer_datetime_format sudah deprecated, gunakan format='mixed'
df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed', errors='coerce')

print(f"\n[2i] Date setelah cleaning:")
print(f"    Range: {df['date'].min()} s/d {df['date'].max()}")
print(f"    Missing: {df['date'].isnull().sum()}")

# ===========================================================
# 📖 BAGIAN 4: Handle Missing Values — Strategi Imputasi
# ===========================================================
# Sekarang semua missing sudah konsisten (NaN). Tinggal pilih strategi:
#   - HAPUS baris: kalau missing sedikit & data banyak
#   - ISI (imputasi): kalau missing banyak, data jangan dibuang
#     * Numerik: isi dengan mean/median
#     * Kategorikal: isi dengan modus (nilai terbanyak)
#   - Biarkan NaN: kalau algoritma ML-nya support (misal: XGBoost)

print("\n" + "=" * 60)
print("STEP 3: HANDLE MISSING VALUES")
print("=" * 60)

print(f"\nMissing sebelum imputasi:")
print(df.isnull().sum())

# Numerik → isi dengan MEDIAN (lebih robust dari mean terhadap outlier)
df['price'] = df['price'].fillna(df['price'].median())
df['quantity'] = df['quantity'].fillna(df['quantity'].median()).astype('Int64')
df['rating'] = df['rating'].fillna(df['rating'].median())

# Kategorikal → isi dengan MODUS (nilai paling sering muncul)
df['city'] = df['city'].fillna(df['city'].mode()[0])
df['category'] = df['category'].fillna(df['category'].mode()[0])

# Tanggal → isi dengan modus (tanggal paling sering)
df['date'] = df['date'].fillna(df['date'].mode()[0])

# Customer name → hapus baris yang kosong (kita butuh tahu siapa customernya)
df = df.dropna(subset=['customer_name'])

print(f"\nMissing setelah imputasi:")
print(df.isnull().sum())

# ===========================================================
# 📖 BAGIAN 5: Feature Engineering Sederhana
# ===========================================================
# Setelah data bersih, kita bisa buat fitur baru yang berguna untuk ML.

print("\n" + "=" * 60)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 60)

# Total harga = price x quantity
df['total_price'] = df['price'] * df['quantity']

# Ekstrak fitur dari tanggal
df['day_of_week'] = df['date'].dt.day_name()            # Senin, Selasa, ...
df['is_weekend'] = df['date'].dt.dayofweek >= 5         # True jika Sabtu/Minggu

print(f"\nKolom baru: total_price, day_of_week, is_weekend")
print(df[['date', 'price', 'quantity', 'total_price', 'day_of_week', 'is_weekend']].head(10))

# ===========================================================
# 📖 BAGIAN 6: Validasi Akhir & Ringkasan
# ===========================================================

print("\n" + "=" * 60)
print("STEP 5: VALIDASI AKHIR")
print("=" * 60)

print(f"\n--- Perbandingan Sebelum vs Sesudah ---")
print(f"{'Metrik':<30} {'Sebelum':<15} {'Sesudah':<15}")
print("-" * 60)
print(f"{'Jumlah baris':<30} {len(raw_data):<15} {len(df):<15}")
print(f"{'Jumlah kolom':<30} {len(raw_data.columns):<15} {len(df.columns):<15}")
print(f"{'Total missing values':<30} {raw_data.isnull().sum().sum():<15} {df.isnull().sum().sum():<15}")
print(f"{'Duplikat':<30} {raw_data.duplicated().sum():<15} {df.duplicated().sum():<15}")

print(f"\n--- Tipe Data Final ---")
print(df.dtypes)

print(f"\n--- Sample Data Bersih ---")
print(df.head(10).to_string())

print("\n" + "=" * 60)
print("CLEANING SELESAI!")
print("=" * 60)
print("""
RANGKUMAN LANGKAH CLEANING:
  1. Inspeksi      → pahami masalah sebelum action
  2. Hapus duplikat → drop_duplicates()
  3. Standardisasi missing → replace variasi kosong ke NaN
  4. Bersihkan teks → strip(), lower(), title(), mapping
  5. Konversi tipe  → to_numeric(), to_datetime()
  6. Tangani outlier → IQR method atau domain knowledge
  7. Imputasi NaN   → median (numerik), mode (kategorikal)
  8. Feature engineering → buat fitur baru dari data bersih
  9. Validasi       → pastikan tidak ada missing, tipe benar

TOOLS PANDAS YANG DIPAKAI:
  - df.isnull().sum()          → cek missing
  - df.duplicated()            → cek duplikat
  - df.drop_duplicates()       → hapus duplikat
  - df.replace()               → ganti nilai
  - df['col'].apply(fn)        → apply fungsi custom
  - pd.to_numeric()            → konversi ke angka
  - pd.to_datetime()           → konversi ke tanggal
  - df['col'].str.strip/lower/title()  → manipulasi string
  - df['col'].map(dict)        → mapping nilai
  - df['col'].fillna()         → isi NaN
  - df.dropna()                → hapus baris NaN
  - df['col'].quantile()       → hitung persentil (untuk IQR)
  - df.describe()              → statistik ringkas

JAWABAN: Ya, Pandas CUKUP untuk data cleaning.
Pandas bahkan TOOL UTAMA yang dipakai data scientist untuk cleaning.
Biasanya 60-80% waktu project ML dihabiskan di tahap ini!
""")
