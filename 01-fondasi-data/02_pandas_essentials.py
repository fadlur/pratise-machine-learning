"""
=============================================================
FASE 1 — MODUL 2: PANDAS ESSENTIALS
=============================================================
Pandas = tool utama untuk data manipulation di ML pipeline.

Kamu perlu Pandas untuk:
- Load & inspect dataset
- Clean data (missing values, outliers, tipe data)
- Transform & feature engineering
- Split data untuk training/testing

Durasi target: 2-3 jam
=============================================================
"""

import numpy as np   # import library NumPy untuk operasi array & numerik
import pandas as pd  # import library Pandas untuk manipulasi data tabular

# ===========================================================
# 📖 BAGIAN 1: Membuat & Membaca Data
# ===========================================================

# Buat DataFrame dari dictionary
# sensor_data mensimulasikan data dari sensor IoT industri dengan 100 baris data
# Kolom:
#   - timestamp  : waktu pencatatan setiap jam selama 100 jam (mulai 2024-01-01)
#   - temperature: suhu dalam °C, distribusi normal (rata-rata=25, std=5)
#   - humidity   : kelembapan dalam %, distribusi normal (rata-rata=60, std=10)
#   - voltage    : tegangan listrik dalam Volt, distribusi normal (rata-rata=220, std=5)
#   - status     : kondisi sensor — 70% normal, 20% warning, 10% fault
sensor_data = pd.DataFrame({                                           # buat DataFrame baru dari dictionary
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),   # buat kolom timestamp: 100 jam berturut-turut mulai 1 Jan 2024
    'temperature': np.random.normal(25, 5, 100),                       # buat 100 nilai suhu acak, rata-rata 25°C, std deviasi 5
    'humidity': np.random.normal(60, 10, 100),                         # buat 100 nilai kelembapan acak, rata-rata 60%, std deviasi 10
    'voltage': np.random.normal(220, 5, 100),                          # buat 100 nilai tegangan acak, rata-rata 220V, std deviasi 5
    'status': np.random.choice(['normal','warning','fault'], 100, p=[0.7, 0.2, 0.1])  # pilih status acak dengan probabilitas 70/20/10%
})

# sengaja tambahkan missing values (realistis!
mask = np.random.random(100) < 0.05                          # buat boolean mask: ~5% baris dipilih secara acak
sensor_data.loc[mask, 'temperature'] = np.nan                # set nilai temperature menjadi NaN pada baris yang terpilih
sensor_data.loc[np.random.random(100) < 0.03, 'voltage'] = np.nan  # set ~3% baris voltage menjadi NaN secara acak

print("📊 Dataset sensor:")                  # cetak judul
print(sensor_data.head(10))                    # tampilkan 10 baris pertama DataFrame
print(f"\nShape: {sensor_data.shape}")          # cetak dimensi DataFrame (baris, kolom)
print(f"\nInfo:")                               # cetak label info
print(sensor_data.info())                      # tampilkan ringkasan tipe data, jumlah non-null, memory usage
print(f"\nStatistik deskriptif:")               # cetak label statistik
print(sensor_data.describe())                  # tampilkan statistik deskriptif (mean, std, min, max, quartiles)

# ===========================================================
# 📖 BAGIAN 2: Data Inspection & Cleaning
# ===========================================================

# Cek missing values
print("\n--- Missing values ---")                       # cetak header bagian missing values
print(sensor_data.isnull().sum())                        # hitung & cetak jumlah NaN per kolom
print(f"Total missing: {sensor_data.isnull().sum().sum()}")  # hitung & cetak total seluruh NaN di DataFrame
# handle missing values - beberapa strategi
# Strategi 1: Drop rows (kehilangan data, tapi simple)
# df_clean = sensor_data.dropna()


# Strategi 2: Fill dengan mean (paling umum untuk numerik)
# df_clean = sensor_data.fillna(sensor_data.mean(numeric_only=True))

# Strategi 3: Interpolasi (BAGUS untuk time series / sensor data!)
df_clean = sensor_data.copy()                                              # buat salinan DataFrame agar data asli tidak berubah
df_clean['temperature'] = df_clean['temperature'].interpolate(method='linear')  # isi NaN temperature dengan interpolasi linier antar nilai tetangga
df_clean['voltage'] = df_clean['voltage'].interpolate(method='linear')          # isi NaN voltage dengan interpolasi linier antar nilai tetangga

print(f"\nSetelah interpolasi, missing: {df_clean.isnull().sum().sum()}")  # cetak sisa NaN setelah interpolasi (seharusnya 0)

# Deteksi outliers menggunakan metode IQR (Interquartile Range)
# Metode ini robust terhadap distribusi non-normal dan umum dipakai di data sensor.
# Logika: nilai yang jatuh di luar rentang [Q1 - 1.5*IQR, Q3 + 1.5*IQR] dianggap outlier.
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)       # kuartil bawah (25% data di bawah nilai ini)
    Q3 = series.quantile(0.75)       # kuartil atas (75% data di bawah nilai ini)
    IQR = Q3 - Q1                    # rentang antar-kuartil = jarak Q3 ke Q1
    lower = Q1 - 1.5 * IQR          # batas bawah: nilai di bawah ini dianggap outlier
    upper = Q3 + 1.5 * IQR          # batas atas: nilai di atas ini dianggap outlier
    return (series < lower) | (series > upper)  # kembalikan boolean mask True = outlier

outlier_mask = detect_outliers_iqr(df_clean['temperature'])  # jalankan deteksi outlier pada kolom temperature, hasilnya boolean mask
print(f"\nOutliers di temperature: {outlier_mask.sum()}")      # cetak jumlah data yang terdeteksi sebagai outlier

# ===========================================================
# 📖 BAGIAN 3: Filtering, Grouping, Aggregation
# ===========================================================

# Filtering
faults = df_clean[df_clean['status'] == 'fault']         # filter hanya baris yang statusnya 'fault'
print(f"\n--- Fault records: {len(faults)} ---")             # cetak jumlah record fault

high_temp_faults = df_clean[                               # filter dengan dua kondisi sekaligus:
    (df_clean['status'] == 'fault') &                      #   status harus 'fault' DAN
    (df_clean['temperature'] > 30)                         #   temperature harus > 30°C
]
print(f"Fault dengan temp > 30°C: {len(high_temp_faults)}")  # cetak jumlah fault yang juga bersuhu tinggi

# Groupby - analisis per kategori
print("\n--- Statistik per Status ---")        # cetak header bagian groupby
grouped = df_clean.groupby('status').agg({    # kelompokkan data berdasarkan kolom 'status', lalu hitung agregasi:
    'temperature': ['mean', 'std', 'min', 'max'],  # untuk temperature: rata-rata, std deviasi, nilai min & max
    'voltage': ['mean', 'std'],                     # untuk voltage: rata-rata dan std deviasi
    'humidity': 'mean'                              # untuk humidity: rata-rata saja
}).round(2)                                    # bulatkan semua hasil ke 2 desimal
print(grouped)                                 # cetak tabel hasil agregasi per status

# Time-based analysis
df_clean['hour'] = df_clean['timestamp'].dt.hour                # ekstrak komponen jam (0-23) dari kolom timestamp ke kolom baru 'hour'
hourly_avg = df_clean.groupby('hour')['temperature'].mean()      # kelompokkan berdasarkan jam, hitung rata-rata temperature per jam
print(f"\n--- Rata-rata suhu per jam (sample) ---")              # cetak header
print(hourly_avg.head())                                         # tampilkan 5 jam pertama sebagai sampel

# ===========================================================
# 📖 BAGIAN 4: Feature Engineering dengan Pandas
# ===========================================================
# Ini KUNCI untuk ML — model hanya sebagus fitur-fiturnya!

# Rolling statistics (moving average, moving std)
df_clean['temp_rolling_mean'] = df_clean['temperature'].rolling(window=5).mean()  # hitung rata-rata bergerak (moving average) dari 5 data terakhir
df_clean['temp_rolling_std'] = df_clean['temperature'].rolling(window=5).std()    # hitung std deviasi bergerak dari 5 data terakhir (ukuran volatilitas)

# Lag features (untuk time series prediction)
df_clean['temp_lag_1'] = df_clean['temperature'].shift(1)  # geser temperature 1 baris ke bawah → nilai temperature 1 langkah sebelumnya
df_clean['temp_lag_3'] = df_clean['temperature'].shift(3)  # geser temperature 3 baris ke bawah → nilai temperature 3 langkah sebelumnya

# Rate of change
df_clean['temp_diff'] = df_clean['temperature'].diff()  # hitung selisih temperature dengan baris sebelumnya (laju perubahan suhu)

# Encode categorical variable
df_clean['status_encoded'] = df_clean['status'].map({  # ubah kolom status (teks) menjadi angka dengan mapping manual:
    'normal': 0, 'warning': 1, 'fault': 2               #   normal→0, warning→1, fault→2 (label encoding)
})

# One-hot encoding
status_dummies = pd.get_dummies(df_clean['status'], prefix='status')  # buat one-hot encoding: setiap nilai unik status jadi kolom biner (0/1)
df_featured = pd.concat([df_clean, status_dummies], axis=1)            # gabungkan kolom one-hot ke DataFrame utama secara horizontal

print("\n--- DataFrame dengan feature baru ---")                        # cetak header
print(df_featured[['temperature', 'temp_rolling_mean', 'temp_lag_1',  # pilih kolom-kolom fitur baru yang sudah dibuat
                   'temp_diff', 'status_encoded']].head(10))          # tampilkan 10 baris pertama sebagai preview


# ===========================================================
# 📖 BAGIAN 5: Persiapan Data untuk ML
# ===========================================================

# Feature matrix (X) dan target (y)
feature_cols = ['temperature', 'humidity', 'voltage',        # daftar nama kolom yang akan dijadikan fitur (input) untuk model ML
                'temp_rolling_mean', 'temp_rolling_std',      # termasuk fitur rolling statistics
                'temp_lag_1', 'temp_lag_3','temp_diff']        # dan fitur lag serta rate of change

# Drop rows dengan NaN (dari rolling/lag features)
df_ml = df_featured.dropna(subset=feature_cols)  # hapus baris yang masih punya NaN di kolom fitur (akibat rolling/lag di awal)

X = df_ml[feature_cols].values                    # ambil kolom fitur dan konversi ke numpy array 2D (matrix fitur)
y = df_ml['status_encoded'].values                # ambil kolom target (status numerik) dan konversi ke numpy array 1D

print(f"\n--- Data siap untuk ML ---")                        # cetak header
print(f"X shape: {X.shape}")                                   # cetak dimensi matrix fitur (jumlah sampel × jumlah fitur)
print(f"y shape: {y.shape}")                                   # cetak dimensi array target (jumlah sampel,)
print(f"Distribusi kelas: {np.bincount(y.astype(int))}")       # hitung & cetak jumlah sampel per kelas (0, 1, 2)

# Train-test split (manual, nanti pakai sklearn)
n = len(X)                                                     # simpan jumlah total sampel
indices = np.random.permutation(n)                             # buat array indeks 0..n-1 yang diacak urutannya (shuffle)
train_size = int(0.8 * n)                                      # hitung ukuran data training = 80% dari total

X_train = X[indices[:train_size]]                              # ambil 80% pertama dari indeks acak sebagai fitur training
X_test = X[indices[train_size:]]                               # ambil 20% sisanya sebagai fitur testing
y_train = y[indices[:train_size]]                              # ambil target training sesuai indeks yang sama
y_test = y[indices[train_size:]]                               # ambil target testing sesuai indeks yang sama

print(f"Train: {X_train.shape}, Test: {X_test.shape}")         # cetak dimensi data train & test untuk verifikasi

# ===========================================================
# 🏋️ EXERCISE 2: Eksplorasi Dataset Publik
# ===========================================================
"""
Download salah satu dataset ini dan lakukan full EDA:

1. Opsi A (EE-related): UCI Power Consumption Dataset
   https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

2. Opsi B (General): Titanic Dataset
   import seaborn as sns
   df = sns.load_dataset('titanic')

Tugas:
a) Load data dan inspect (shape, dtypes, missing values)
b) Bersihkan data (handle missing, outliers)
c) Buat minimal 5 fitur baru yang meaningful
d) Visualisasi (di modul selanjutnya, tapi coba dulu)
e) Siapkan X dan y untuk ML

PENTING: Tulis INSIGHT, bukan cuma kode!
Contoh insight: "Voltage drop > 10V berkorelasi dengan status 'fault' —
ini masuk akal karena fault biasanya menyebabkan voltage sag."
"""

import seaborn as sns  # import seaborn untuk akses dataset bawaan (Titanic)

print("\n" + "="*50)
print("🏋️ EXERCISE 2: EDA Dataset Titanic")
print("="*50)

# --- a) Load data dan inspect ---
df = sns.load_dataset('titanic')  # load dataset Titanic dari seaborn (891 penumpang, 15 kolom)

print("\n📌 a) Inspect Data")
print(f"Shape: {df.shape}")                    # cetak dimensi: (891, 15)
print(f"\nTipe data:\n{df.dtypes}")            # cetak tipe data setiap kolom
print(f"\nMissing values:\n{df.isnull().sum()}")  # cetak jumlah NaN per kolom
print(f"\nTotal missing: {df.isnull().sum().sum()}")  # total NaN keseluruhan
print(f"\nSample data:\n{df.head()}")          # tampilkan 5 baris pertama
print(f"\nStatistik deskriptif:\n{df.describe()}")    # statistik numerik
print(f"\nDistribusi target (survived):\n{df['survived'].value_counts()}")  # cek balance kelas

# INSIGHT: Dataset punya 891 penumpang. Kolom 'age' punya ~177 NaN (19.9%),
# 'deck' punya 688 NaN (77.2%) — terlalu banyak, sebaiknya di-drop.
# 'embarked' dan 'embark_town' hanya 2 NaN — bisa diisi dengan modus.
# Target 'survived' cukup imbalanced: ~38% survived, ~62% tidak.

# --- b) Bersihkan data ---
print("\n📌 b) Data Cleaning")
df_titanic = df.copy()  # buat salinan agar data asli tetap utuh

# Drop kolom yang terlalu banyak missing atau redundan
df_titanic = df_titanic.drop(columns=['deck', 'embark_town', 'alive', 'who', 'adult_male'])
# deck: 77% NaN, tidak berguna
# embark_town: duplikat dari 'embarked'
# alive: duplikat dari 'survived'
# who: duplikat dari 'sex' + 'age'
# adult_male: redundan dengan 'sex' + 'age'

# Isi missing 'age' dengan median per kelas (lebih akurat daripada median global)
df_titanic['age'] = df_titanic.groupby('pclass')['age'].transform(
    lambda x: x.fillna(x.median())  # isi NaN age dengan median age di kelas yang sama
)

# Isi missing 'embarked' dengan modus (nilai paling sering)
df_titanic['embarked'] = df_titanic['embarked'].fillna(
    df_titanic['embarked'].mode()[0]  # ambil nilai modus (S = Southampton)
)

print(f"Missing setelah cleaning:\n{df_titanic.isnull().sum()}")
print(f"Total missing: {df_titanic.isnull().sum().sum()}")

# INSIGHT: Mengisi 'age' berdasarkan median per kelas penumpang lebih baik
# daripada median global, karena penumpang kelas 1 cenderung lebih tua
# (median ~37) dibanding kelas 3 (median ~24).

# Deteksi outlier pada 'fare' menggunakan IQR
fare_outliers = detect_outliers_iqr(df_titanic['fare'])  # reuse fungsi dari bagian sebelumnya
print(f"\nOutliers di fare: {fare_outliers.sum()}")
# INSIGHT: Ada cukup banyak outlier di 'fare' — wajar karena tiket kelas 1
# bisa sangat mahal (max ~$512) sementara kelas 3 sangat murah (~$7).
# Kita biarkan outlier ini karena harga tiket yang sangat tinggi adalah data valid,
# bukan error pengukuran.

# --- c) Buat minimal 5 fitur baru yang meaningful ---
print("\n📌 c) Feature Engineering (5+ fitur baru)")

# Fitur 1: family_size — total anggota keluarga di kapal
df_titanic['family_size'] = df_titanic['sibsp'] + df_titanic['parch'] + 1
# sibsp = jumlah saudara/pasangan, parch = jumlah orang tua/anak
# +1 untuk menghitung diri sendiri

# Fitur 2: is_alone — apakah penumpang sendirian (tanpa keluarga)
df_titanic['is_alone'] = (df_titanic['family_size'] == 1).astype(int)
# 1 jika sendirian, 0 jika punya keluarga

# Fitur 3: age_group — kategorisasi umur
df_titanic['age_group'] = pd.cut(df_titanic['age'],
    bins=[0, 12, 18, 35, 60, 100],                       # batas kelompok umur
    labels=['child', 'teen', 'young_adult', 'adult', 'senior']  # label kategori
)
# INSIGHT: Anak-anak (child) punya survival rate lebih tinggi karena
# kebijakan "women and children first".

# Fitur 4: fare_per_person — harga tiket per orang (bagi fare dengan family_size)
df_titanic['fare_per_person'] = df_titanic['fare'] / df_titanic['family_size']
# Beberapa tiket dibeli untuk satu keluarga, jadi fare/person lebih representatif

# Fitur 5: fare_category — kategorisasi harga tiket (murah/sedang/mahal)
df_titanic['fare_category'] = pd.qcut(df_titanic['fare'],
    q=3,                                                  # bagi menjadi 3 kuantil (33% masing-masing)
    labels=['cheap', 'medium', 'expensive']               # label: murah, sedang, mahal
)
# INSIGHT: Fare mencerminkan kelas sosial — penumpang dengan tiket mahal
# umumnya kelas 1 dan punya akses lebih baik ke sekoci.

# Fitur 6: cabin_known — apakah info kabin (deck) diketahui (proxy untuk status sosial)
df_titanic['cabin_known'] = df['deck'].notna().astype(int)  # pakai data asli 'df' yang masih punya kolom 'deck'
# 1 jika deck tercatat, 0 jika tidak — penumpang kelas tinggi lebih sering punya data kabin

print(f"Fitur baru yang dibuat:")
print(f"  - family_size: {df_titanic['family_size'].describe().round(2).to_dict()}")
print(f"  - is_alone: {df_titanic['is_alone'].value_counts().to_dict()}")
print(f"  - age_group: {df_titanic['age_group'].value_counts().to_dict()}")
print(f"  - fare_per_person mean: {df_titanic['fare_per_person'].mean():.2f}")
print(f"  - fare_category: {df_titanic['fare_category'].value_counts().to_dict()}")
print(f"  - cabin_known: {df_titanic['cabin_known'].value_counts().to_dict()}")

# Survival rate per fitur baru
print(f"\nSurvival rate per group:")
print(f"  is_alone: {df_titanic.groupby('is_alone')['survived'].mean().round(3).to_dict()}")
print(f"  age_group: {df_titanic.groupby('age_group')['survived'].mean().round(3).to_dict()}")
print(f"  fare_category: {df_titanic.groupby('fare_category')['survived'].mean().round(3).to_dict()}")
print(f"  cabin_known: {df_titanic.groupby('cabin_known')['survived'].mean().round(3).to_dict()}")

# INSIGHT: Penumpang yang sendirian (is_alone=1) punya survival rate ~30%,
# sedangkan yang punya keluarga ~50%. Penumpang dengan kabin tercatat
# punya survival rate jauh lebih tinggi — mereka cenderung kelas 1/2.

# --- d) & e) Siapkan X dan y untuk ML ---
print("\n📌 e) Persiapan Data untuk ML")

# Encode categorical columns
df_titanic_ml = df_titanic.copy()

# Label encode 'sex'
df_titanic_ml['sex'] = df_titanic_ml['sex'].map({'male': 0, 'female': 1})  # male→0, female→1

# Label encode 'embarked'
df_titanic_ml['embarked'] = df_titanic_ml['embarked'].map({'S': 0, 'C': 1, 'Q': 2})  # S→0, C→1, Q→2

# Label encode 'fare_category'
fare_cat_map = {'cheap': 0, 'medium': 1, 'expensive': 2}  # mapping harga: murah→0, sedang→1, mahal→2
df_titanic_ml['fare_category_encoded'] = df_titanic_ml['fare_category'].map(fare_cat_map)

# Label encode 'age_group'
age_group_map = {'child': 0, 'teen': 1, 'young_adult': 2, 'adult': 3, 'senior': 4}
df_titanic_ml['age_group_encoded'] = df_titanic_ml['age_group'].map(age_group_map)

# Pilih kolom fitur untuk ML
titanic_feature_cols = [
    'pclass', 'sex', 'age', 'fare', 'embarked',           # fitur asli
    'family_size', 'is_alone', 'fare_per_person',           # fitur buatan
    'fare_category_encoded', 'age_group_encoded', 'cabin_known'  # fitur encoded
]

# Drop baris dengan NaN (jika ada sisa)
df_titanic_ml = df_titanic_ml.dropna(subset=titanic_feature_cols)

X_titanic = df_titanic_ml[titanic_feature_cols].values  # matrix fitur (numpy array 2D)
y_titanic = df_titanic_ml['survived'].values            # target: 0 = meninggal, 1 = selamat

print(f"X shape: {X_titanic.shape}")         # dimensi fitur
print(f"y shape: {y_titanic.shape}")         # dimensi target
print(f"Survival rate: {y_titanic.mean():.3f}")  # proporsi yang selamat
print(f"Fitur: {titanic_feature_cols}")      # daftar nama fitur yang dipakai

# Train-test split (manual)
n_titanic = len(X_titanic)
idx_titanic = np.random.permutation(n_titanic)        # acak indeks
train_sz = int(0.8 * n_titanic)                       # 80% untuk training

X_train_t = X_titanic[idx_titanic[:train_sz]]         # fitur training
X_test_t = X_titanic[idx_titanic[train_sz:]]          # fitur testing
y_train_t = y_titanic[idx_titanic[:train_sz]]         # target training
y_test_t = y_titanic[idx_titanic[train_sz:]]          # target testing

print(f"Train: {X_train_t.shape}, Test: {X_test_t.shape}")

# INSIGHT KESELURUHAN:
# Faktor paling berpengaruh terhadap survival di Titanic adalah:
# 1. Sex (female survival rate ~74% vs male ~19%) — "women first"
# 2. Pclass (kelas 1: ~63%, kelas 3: ~24%) — akses ke sekoci
# 3. Age (anak-anak punya peluang lebih tinggi) — "children first"
# 4. Fare/cabin — proxy untuk status sosial ekonomi


# ===========================================================
# 🔥 CHALLENGE: Pipeline Otomatis
# ===========================================================
"""
Buat class DataPipeline yang:
1. __init__(self, df) — terima raw DataFrame
2. .inspect() — print summary (shape, missing, dtypes)
3. .clean(strategy='interpolate') — handle missing values
4. .add_rolling_features(columns, windows) — tambah rolling stats
5. .add_lag_features(columns, lags) — tambah lag features
6. .encode_categorical(columns) — encode categorical columns
7. .prepare_ml(target_col, feature_cols) — return X_train, X_test, y_train, y_test

Class ini akan berguna di semua project selanjutnya!

Kenapa bikin class sendiri? Karena di dunia nyata, data pipeline = 80% waktu ML.
Lebih baik punya pipeline yang solid daripada model yang fancy.
"""


class DataPipeline:
    """Pipeline otomatis untuk preprocessing data sebelum masuk ke model ML."""

    def __init__(self, df):
        """Terima raw DataFrame dan simpan sebagai salinan internal."""
        self.df = df.copy()      # simpan salinan agar DataFrame asli tidak berubah
        self.original = df.copy() # simpan backup data original untuk referensi

    def inspect(self):
        """Print ringkasan dataset: shape, missing values, tipe data, statistik."""
        print("="*50)
        print("📋 DATA INSPECTION")
        print("="*50)
        print(f"Shape: {self.df.shape}")                            # dimensi (baris, kolom)
        print(f"\nTipe data:\n{self.df.dtypes}")                    # tipe data per kolom
        print(f"\nMissing values:\n{self.df.isnull().sum()}")       # jumlah NaN per kolom
        print(f"Total missing: {self.df.isnull().sum().sum()}")     # total NaN
        print(f"\nStatistik deskriptif:\n{self.df.describe()}")     # statistik numerik
        return self  # return self untuk method chaining

    def clean(self, strategy='interpolate'):
        """Handle missing values dengan strategi yang dipilih.

        Args:
            strategy: 'interpolate' (default), 'mean', 'median', atau 'drop'
        """
        print(f"\n🧹 Cleaning data dengan strategi: {strategy}")
        before = self.df.isnull().sum().sum()  # hitung NaN sebelum cleaning

        if strategy == 'interpolate':
            # Interpolasi linier — cocok untuk time series / data berurutan
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns  # ambil kolom numerik saja
            for col in numeric_cols:
                self.df[col] = self.df[col].interpolate(method='linear')  # isi NaN dengan interpolasi linier
        elif strategy == 'mean':
            # Isi NaN dengan rata-rata kolom — simple, tapi bisa bias oleh outlier
            self.df = self.df.fillna(self.df.mean(numeric_only=True))
        elif strategy == 'median':
            # Isi NaN dengan median — lebih robust terhadap outlier daripada mean
            self.df = self.df.fillna(self.df.median(numeric_only=True))
        elif strategy == 'drop':
            # Hapus semua baris yang mengandung NaN — simple tapi kehilangan data
            self.df = self.df.dropna()

        after = self.df.isnull().sum().sum()  # hitung NaN setelah cleaning
        print(f"Missing: {before} → {after}")
        return self

    def add_rolling_features(self, columns, windows):
        """Tambah fitur rolling statistics (mean & std) untuk kolom dan window tertentu.

        Args:
            columns: list nama kolom numerik
            windows: list ukuran window (misal [3, 5, 10])
        """
        print(f"\n📊 Menambah rolling features...")
        for col in columns:                           # loop setiap kolom
            for w in windows:                         # loop setiap ukuran window
                self.df[f'{col}_rolling_mean_{w}'] = (
                    self.df[col].rolling(window=w).mean()  # rata-rata bergerak
                )
                self.df[f'{col}_rolling_std_{w}'] = (
                    self.df[col].rolling(window=w).std()   # std deviasi bergerak
                )
                print(f"  + {col}_rolling_mean_{w}, {col}_rolling_std_{w}")
        return self

    def add_lag_features(self, columns, lags):
        """Tambah fitur lag (nilai X langkah sebelumnya) untuk time series.

        Args:
            columns: list nama kolom
            lags: list jumlah lag (misal [1, 3, 5])
        """
        print(f"\n⏪ Menambah lag features...")
        for col in columns:                           # loop setiap kolom
            for lag in lags:                          # loop setiap nilai lag
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)  # geser data ke bawah sebanyak lag
                print(f"  + {col}_lag_{lag}")
        return self

    def encode_categorical(self, columns):
        """Encode kolom kategorikal menjadi numerik menggunakan label encoding.

        Args:
            columns: list nama kolom kategorikal
        """
        print(f"\n🔢 Encoding categorical columns...")
        for col in columns:
            unique_vals = self.df[col].unique()       # ambil nilai unik di kolom
            mapping = {val: idx for idx, val in enumerate(unique_vals) if pd.notna(val)}
            # buat mapping otomatis: setiap nilai unik → angka 0, 1, 2, ...
            self.df[f'{col}_encoded'] = self.df[col].map(mapping)  # terapkan mapping
            print(f"  + {col}_encoded: {mapping}")
        return self

    def prepare_ml(self, target_col, feature_cols, test_size=0.2):
        """Siapkan data untuk ML: buat X (fitur) dan y (target), lalu split train/test.

        Args:
            target_col: nama kolom target
            feature_cols: list nama kolom fitur
            test_size: proporsi data test (default 0.2 = 20%)

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"\n🎯 Menyiapkan data ML...")
        df_ready = self.df.dropna(subset=feature_cols + [target_col])  # hapus baris dengan NaN di kolom penting
        print(f"Baris setelah drop NaN: {len(df_ready)} dari {len(self.df)}")

        X = df_ready[feature_cols].values   # matrix fitur sebagai numpy array
        y = df_ready[target_col].values     # array target sebagai numpy array

        # Shuffle dan split
        n = len(X)
        indices = np.random.permutation(n)            # acak urutan indeks
        split = int((1 - test_size) * n)              # hitung titik split

        X_train = X[indices[:split]]                  # fitur training
        X_test = X[indices[split:]]                   # fitur testing
        y_train = y[indices[:split]]                  # target training
        y_test = y[indices[split:]]                   # target testing

        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
        return X_train, X_test, y_train, y_test


# --- Demo: Gunakan DataPipeline pada sensor_data ---
print("\n" + "="*50)
print("🔥 CHALLENGE: Demo DataPipeline pada sensor_data")
print("="*50)

pipeline = DataPipeline(sensor_data)                  # inisialisasi pipeline dengan data sensor mentah

pipeline.inspect()                                     # tampilkan ringkasan data

pipeline.clean(strategy='interpolate')                # bersihkan missing values dengan interpolasi

pipeline.add_rolling_features(                        # tambah rolling features
    columns=['temperature', 'voltage'],               #   untuk kolom temperature & voltage
    windows=[3, 5]                                    #   dengan window 3 dan 5
)

pipeline.add_lag_features(                            # tambah lag features
    columns=['temperature'],                          #   untuk kolom temperature
    lags=[1, 3]                                       #   lag 1 dan 3 langkah
)

pipeline.encode_categorical(columns=['status'])       # encode kolom status ke numerik

# Siapkan data ML
ml_features = [
    'temperature', 'humidity', 'voltage',
    'temperature_rolling_mean_5', 'temperature_rolling_std_5',
    'temperature_lag_1', 'temperature_lag_3'
]

X_tr, X_te, y_tr, y_te = pipeline.prepare_ml(        # jalankan prepare_ml
    target_col='status_encoded',                      # target: status sensor (numerik)
    feature_cols=ml_features                          # fitur yang dipilih
)

print(f"\n✅ Pipeline selesai! Data siap untuk training model.")

print("\n" + "="*50)
print("✅ Modul 2 selesai! Lanjut ke: 01-fondasi-data/03_visualisasi.py")
print("="*50)