"""
=============================================================
FASE 1 — MODUL 1: NUMPY ESSENTIALS
=============================================================
Kenapa NumPy dulu?
- Semua library ML di Python dibangun di atas NumPy
- Memahami array operations = memahami cara kerja internal ML
- Dengan background Teknik Elektro, kamu sudah paham matrix algebra
  → ini tinggal mapping ke sintaks Python

Durasi target: 2-3 jam
=============================================================
"""
import numpy as np
# ===========================================================
# 📖 BAGIAN 1: Array Creation & Basic Operations
# ===========================================================
# Sebagai engineer, kamu sudah familiar dengan vektor dan matrix.
# NumPy array = representasi efisien dari struktur data ini.

# Membuat array dari list

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Vektor x: {x}")
print(f"Shape: {x.shape}, Dtype: {x.dtype}")

# Matrix 2D
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(f"\nMatrix A:\n{A}")
print(f"Shape: {A.shape}")

# Array generators - sering dipakai untuk inisialisasi
zeros = np.zeros((3, 4))        # matrix nol
print(f"\nMatrix zero:\n {zeros}")
ones = np.ones((2, 3))          # matrix satu
print(f"\nMatrix one:\n {ones}")
identity = np.eye(4)            # identity matrix (familiar dari linear algebra)
# angka 1 diagonal di matrix 4 x 4
print(f"\nMatrix eye:\n {identity}")
random_normal = np.random.randn(3, 3) # random dari distribusi normal
print(f"Random normal:\n {random_normal}")
# Linspace - familiar dari MATLAB/signal processing
t = np.linspace(0, 2 * np.pi, 100)  # 100 titik yang terbagi rata dari 0 sampai 2π
signal = np.sin(t)    # sinyal sinusoisal - kamu pasti sering pakai ini

print(f"\nSinyal sinusoidal: {signal[:5]}...")  # 5 sample pertama

# ===========================================================
# 📖 BAGIAN 2: Broadcasting & Vectorization
# ===========================================================
# INI KUNCI PENTING untuk ML!
# Broadcasting = operasi antara array dengan shape berbeda
# Vectorization = hindari loop, gunakan operasi array

# Contoh: normalisasi data (z-score)
# Rumus: z = (x - mean) / std

data = np.random.randn(1000, 5)   # 1000 samples, 5 features
print(f"Data: \n {data}")

# CARA BURUK (loop) - jangan lakukan ini
# for i in range(data.shape[1]):
#     data[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].std()

# CARA BAIK (vectorized + broadcasting)
mean = data.mean(axis=0)  # mean per kolom → shape (5,)
std = data.std(axis=0)    # standard deviasi - std per kolom → shape (5,)
data_normalized = (data - mean) / std   # broadcasting: (1000, 5) - (5,) → otomatis!

# data_normalized = (data - mean) / std
print(f"\nNormalized mean (harus ~0): {data_normalized.mean(axis=0).round(4)}")
print(f"Normalized std (harus ~1): {data_normalized.std(axis=0).round(4)}")

# ===========================================================
# 📖 BAGIAN 3: Linear Algebra Operations
# ===========================================================
# Ini yang paling relevan untuk ML. Hampir semua model ML
# pada dasarnya adalah operasi linear algebra.

A = np.random.randn(3, 4)
B = np.random.randn(4, 2)

print(f"Matrix A:\n {A}")
print(f"\nMatrix B:\n {B}")
# Matrix multiplication
C = A @ B # atau np.dot(A, B)
print(f"\nMatrix C:\n {C}")
print(f"\nMatrix multiplication: ({A.shape}) @ ({B.shape}) = {C.shape}")

# Transpose (kolom jadi baris atau sebaliknya)
print(f"A^T shape: {A.T.shape}")

# Eigenvalue decomposition - pasti familiar dari kuliah
M = np.random.randn(3, 3)
M = M @ M.T # buat symmetric position definite
eigenvalues, eigenvectors = np.linalg.eigh(M)
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvectors: {eigenvectors}")