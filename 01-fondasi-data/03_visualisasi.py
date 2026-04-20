"""
=============================================================
FASE 1 — MODUL 3: VISUALISASI DATA
=============================================================
"If you can't see it, you can't understand it."

Visualisasi bukan cuma untuk presentasi — ini tool DEBUGGING
paling powerful di ML. Kamu HARUS bisa visualisasi:
1. Distribusi data (histogram, KDE)
2. Relasi antar fitur (scatter, correlation)
3. Model performance (learning curves, confusion matrix)

Durasi target: 2-3 jam
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ===========================================================
# 📖 BAGIAN 1: Distribusi Data
# ===========================================================
# TUJUAN: Memahami bentuk distribusi data.
#   Kenapa penting? Banyak algoritma ML (misal: Linear Regression, Naive Bayes)
#   mengasumsikan data berdistribusi normal. Kalau data skewed atau bimodal,
#   kita perlu tahu supaya bisa preprocessing (scaling, transformasi) dengan benar.
#
# YANG DIPELAJARI:
#   - Histogram: melihat sebaran frekuensi data
#   - Mean vs Median: jika keduanya berbeda jauh → distribusi skewed
#   - 3 jenis distribusi: normal (simetris), skewed (miring), bimodal (2 puncak)

# seed(42) = memastikan angka random selalu sama setiap kali dijalankan (reproducible)
np.random.seed(42)
data = pd.DataFrame({
    'normal': np.random.normal(0, 1, 1000),         # distribusi simetris, mean ≈ median
    'skewed': np.random.exponential(2, 1000),        # distribusi miring ke kanan, mean > median
    'bimodal': np.concatenate([                      # distribusi 2 puncak (campuran 2 normal)
        np.random.normal(-2, 0.5, 500),
        np.random.normal(2, 0.5, 500)
    ])
})

# Buat 3 subplot berdampingan (1 baris, 3 kolom)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, col in enumerate(data.columns):
    axes[i].hist(data[col], bins=50, alpha=0.7, edgecolor='black')  # histogram 50 bin
    axes[i].set_title(f'Distribusi: {col}')
    axes[i].axvline(data[col].mean(), color='red', linestyle='--', label='mean')    # garis mean
    axes[i].axvline(data[col].median(), color='green', linestyle='--', label='median')  # garis median
    axes[i].legend()

plt.tight_layout()
plt.savefig('01_distribusi.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 01_distribusi.png")

# INSIGHT: kalau mean ≠ median → distribusi skewed
# ini penting karena banyak model ML mengasumsikan distribusi normal

# ===========================================================
# 📖 BAGIAN 2: Relasi Antar Fitur
# ===========================================================
# TUJUAN: Melihat hubungan (korelasi) antar fitur dalam dataset.
#   Kenapa penting?
#   - Fitur yang sangat berkorelasi satu sama lain → redundan (multicollinearity)
#     dan bisa bikin model tidak stabil.
#   - Fitur yang tidak berkorelasi dengan target → kemungkinan tidak berguna.
#   - Correlation heatmap = cara cepat melihat semua korelasi sekaligus.
#   - Scatter matrix = melihat pola hubungan secara visual per pasangan fitur.

# Buat dataset sintetis dengan korelasi yang sengaja diatur
n = 200
X1 = np.random.randn(n)                              # fitur dasar (acak)
X2 = 0.5 * X1 + np.random.randn(n) * 0.5             # korelasi POSITIF dengan X1
X3 = -0.8 * X1 + np.random.randn(n) * 0.3             # korelasi NEGATIF dengan X1
X4 = np.random.randn(n)                               # TIDAK berkorelasi dengan X1
# Target: label biner (0 atau 1), ditentukan oleh kombinasi X1, X2, X3 + noise
y = (X1 + X2 - X3 + np.random.randn(n) * 0.5 > 0).astype(int)

df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'target': y})

# --- Correlation Heatmap ---
# corr() menghitung korelasi Pearson antar semua kolom (-1 s/d +1)
# Nilai mendekati +1 = korelasi positif kuat, -1 = negatif kuat, 0 = tidak ada korelasi
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title('Correlation Matrix')

# --- Scatter Matrix ---
# Menampilkan scatter plot untuk setiap pasangan fitur
# Diagonal = distribusi masing-masing fitur, off-diagonal = relasi antar 2 fitur
# Warna titik berdasarkan label target (y) → bisa lihat apakah kelas terpisah
pd.plotting.scatter_matrix(df[['X1', 'X2', 'X3', 'X4']],
                           c=y, cmap='RdYlBu', alpha=0.5,
                           figsize=(10, 10))
plt.savefig('02_sactter_matrix.png', dpi=100, bbox_inches='tight')
plt.close()

# Simpan heatmap korelasi sebagai file terpisah
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Correlation Matrix")
plt.tight_layout()
plt.savefig("02_correlation.png", dpi=100, bbox_inches="tight")
plt.close()
print("📊 Saved: 02_correlation.png, 02_scatter_matrix.png")


# ===========================================================
# 📖 BAGIAN 3: Visualisasi untuk ML
# ===========================================================
# TUJUAN: Memvisualisasi "decision boundary" — garis/area yang menunjukkan
#   bagaimana model memisahkan kelas di ruang 2D.
#   Kenapa penting?
#   - Kita bisa MELIHAT bagaimana model mengambil keputusan
#   - Berguna untuk membandingkan model linear vs non-linear
#   - Tool debugging paling visual untuk klasifikasi

from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model_predict_fn, title="Decision Boundary"):
    """Visualisasi decision boundary - akan sering dipakai nanti"""
    h = 0.02  # resolusi grid (makin kecil = makin halus tapi makin lambat)
    # Tentukan batas area plot berdasarkan range data + margin
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Buat grid titik-titik yang menutupi seluruh area
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Prediksi kelas untuk SETIAP titik di grid → menghasilkan "peta" keputusan
    Z = model_predict_fn(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")  # area warna = keputusan model
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="black", s=50)  # titik data asli
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    return plt

# Contoh: linear decision boundary
# Ambil 2 fitur (X1, X2) supaya bisa diplot di 2D
X_2d = df[['X1', 'X2']].values

# Classifier sederhana: prediksi kelas 1 jika X1 + X2 > 0, else kelas 0
# Ini membuat garis lurus diagonal sebagai batas keputusan
def simple_linear_classifier(X):
    return (X[:, 0] + X[:, 1] > 0).astype(int)


fig = plot_decision_boundary(X_2d, y, simple_linear_classifier,
                             "Contoh: Linear Decision Boundary")

plt.savefig("03_decision_boundary.png", dpi=100, bbox_inches="tight")
plt.close()
print("📊 Saved: 03_decision_boundary.png")

# ===========================================================
# 📖 BAGIAN 4: Visualisasi Model Performance
# ===========================================================
# TUJUAN: Mengukur dan memvisualisasi seberapa bagus performa model.
#   Dua tool utama:
#   1. Confusion Matrix → melihat detail prediksi benar/salah per kelas
#   2. Learning Curve → mendiagnosa overfitting vs underfitting

# --- Confusion Matrix ---
# Confusion matrix menunjukkan:
#   - True Positive (TP): prediksi 1, kenyataan 1 ✓
#   - True Negative (TN): prediksi 0, kenyataan 0 ✓
#   - False Positive (FP): prediksi 1, kenyataan 0 ✗ (Type I error)
#   - False Negative (FN): prediksi 0, kenyataan 1 ✗ (Type II error)
def plot_confusion_matrix(y_true, y_pred, classes=['Class 0', 'Class 1']):
    """"Plot confusion matrix - Wajib tahu untuk evaluasi model"""
    # Hitung confusion matrix secara manual (tanpa sklearn)
    n_classes=len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):  # t=label asli, p=prediksi model
        cm[t][p] += 1  # cm[baris=actual][kolom=predicted]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    
    ax.set_ylabel("Actual")     # baris = label yang sebenarnya
    ax.set_xlabel("Predicted")  # kolom = prediksi model
    ax.set_title("Confusion Matrix")
    return fig

# Contoh: 50 data kelas 0, 50 data kelas 1
# Model salah prediksi: 5 kelas 0 diprediksi 1 (FP), 10 kelas 1 diprediksi 0 (FN)
y_true = np.array([0]*50 + [1]*50)
y_pred = np.array([0]*45 + [1]*5 + [0]*10 + [1]*40)
fig = plot_confusion_matrix(y_true, y_pred)
plt.savefig("04_confusion_matrix.png", dpi=100, bbox_inches="tight")
plt.close()
print("📊 Saved: 04_confusion_matrix.png")

# --- Learning Curve ---
# Memplot skor training vs validation terhadap jumlah data training.
# Cara baca:
#   - Training score TURUN seiring data bertambah → normal (model tidak bisa hafal semua data)
#   - Validation score NAIK seiring data bertambah → model makin generalize
#   - GAP BESAR antara train & val → OVERFITTING (model hafal training, gagal di data baru)
#   - KEDUANYA RENDAH → UNDERFITTING (model terlalu sederhana)
#   - fill_between = area confidence band (ketidakpastian skor)
def plot_learning_curve(train_sizes, train_scores, val_scores, title="Learning curve"):
    """Plot learning curve - untuk diagnosa overfitting/underfitting"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores, "o-", label="Training Score")
    ax.plot(train_sizes, val_scores, "o-", label="Validation Score")
    ax.fill_between(train_sizes,
                    train_scores - 0.02, train_scores + 0.02, alpha=0.1)
    ax.fill_between(train_sizes,
                    val_scores - 0.05, val_scores + 0.05, alpha=0.1)
    ax.set_xlabel("Training Size")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    return fig

# Contoh data simulasi learning curve:
# train_acc turun dari 0.99 → 0.92 (model makin tidak bisa hafal semua)
# val_acc naik dari 0.70 → 0.89 (model makin bagus di data baru)
# Gap mengecil → tanda model mulai seimbang (less overfitting)
sizes = np.array([50, 100, 200, 400, 800])
train_acc = np.array([0.99, 0.97, 0.95, 0.93, 0.92]) # menurun (less overfit)
val_acc = np.array([0.70, 0.78, 0.83, 0.87, 0.89]) # meningkat

fig = plot_learning_curve(sizes, train_acc, val_acc)
plt.savefig("05_learning_curve.png", dpi=100, bbox_inches="tight")
plt.close()
print("📊 Saved: 05_learning_curve.png")

# ===========================================================
# 📖 BAGIAN 5: Signal Processing Visualization (EE-Relevant!)
# ===========================================================
# TUJUAN: Menunjukkan koneksi antara signal processing (Electrical Engineering)
#   dengan Machine Learning.
#   Kenapa relevan?
#   - Banyak aplikasi ML bekerja dengan data sinyal: audio, getaran (vibration),
#     sensor akselerometer, EEG, ECG, dll.
#   - Dari sinyal, kita bisa ekstrak FITUR untuk ML:
#     * Time domain: mean, std, peak, RMS
#     * Frequency domain: dominant frequency, spectral energy, bandwidth
#   - FFT (Fast Fourier Transform) mengubah sinyal dari domain waktu → frekuensi

# Sinyal + FFT — familiar territory!
fs = 1000  # sampling frequency 1kHz (1000 sampel per detik)
t = np.arange(0, 1, 1/fs)  # array waktu: 0 sampai 1 detik, step 1ms → 1000 titik

# Buat sinyal campuran dari beberapa komponen:
signal = (np.sin(2 * np.pi * 50 * t) +       # komponen 50 Hz (amplitudo 1.0)
          0.5 * np.sin(2 * np.pi * 120 * 5) + # BUG: harusnya '120 * t' bukan '120 * 5'
          0.3 * np.random.randn(len(t)))       # noise acak (Gaussian)
# NOTE: baris ke-2 menghasilkan konstanta karena pakai angka 5, bukan variabel t
#       sehingga komponen 120 Hz tidak muncul di FFT. Fix: ganti 5 → t

# FFT: mengubah sinyal dari domain waktu → domain frekuensi
# Hasilnya menunjukkan komponen frekuensi apa saja yang ada di sinyal
fft_vals = np.fft.fft(signal)              # hitung FFT
freqs = np.fft.fftfreq(len(t), 1/fs)      # array frekuensi yang bersesuaian

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Plot domain waktu — sinyal seperti yang terlihat di osiloskop
axes[0].plot(t[:200], signal[:200])  # tampilkan 200 sampel pertama (= 200ms)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Signal (Time Domain)")

# Plot domain frekuensi — menunjukkan komponen frekuensi (seperti spectrum analyzer)
# Ambil setengah pertama saja (frekuensi positif, karena FFT simetris)
positive_freqs = freqs[:len(freqs)//2]
magnitude = np.abs(fft_vals[:len(fft_vals)//2]) * 2/len(t)  # normalisasi amplitudo
axes[1].plot(positive_freqs, magnitude)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude")
axes[1].set_title("Signal (Frequency Domain)")
axes[1].set_xlim(0, 200)  # tampilkan 0-200 Hz saja

plt.tight_layout()
plt.savefig("06_signal_analysis.png", dpi=100, bbox_inches="tight")
plt.close()
print("📊 Saved: 06_signal_analysis.png")

# KONEKSI KE ML:
# - Time domain features → statistik (mean, std, peak)
# - Frequency domain features → dominant frequency, spectral energy
# - Ini PERSIS yang dilakukan di audio/vibration ML!