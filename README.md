# Tugas Besar 1 IF3270 Pembelajaran Mesin

## Feedforward Neural Network (FFNN) from Scratch

### **📌 Deskripsi Singkat**
Tugas ini mengimplementasikan **Feedforward Neural Network (FFNN)** dari awal tanpa menggunakan library deep learning seperti TensorFlow atau PyTorch. Model ini mendukung berbagai fungsi aktivasi, fungsi loss, dan mekanisme training menggunakan **gradient descent**.

---

## **📁 Struktur Repository**

```
📦 TugasBesar1_FFNN
 ┣ 📂 src
 ┃ ┣ 📜 ffnn.py  # Implementasi kelas FFNN
 ┃ ┣ 📜 train.py  # Script untuk melatih model
 ┃ ┣ 📜 test.ipynb  # Notebook pengujian model
 ┃ ┣ 📜 uv.lock  # Daftar dependency yang dikelola oleh uv
 ┃ ┗ 📜 utils.py  # Fungsi tambahan (visualisasi, preprocessing)
 ┣ 📂 doc
 ┃ ┣ 📜 laporan.pdf  # Laporan tugas besar
 ┃ ┗ 📜 references.bib  # Daftar referensi
 ┗ 📜 README.md  # Dokumentasi proyek
```

---

## **🚀 Cara Menjalankan Program**
### **1️⃣ Instalasi Dependency dengan uv dan venv**
```bash
# Masuk ke src
cd src

# Buat virtual environment dengan uv
uv venv venv

# Aktifkan virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Instal dependency dari uv.lock
uv pip sync
```

### **2️⃣ Menjalankan Model**
```bash
python train.py
```

### **3️⃣ Menguji Model**
Jalankan file notebook:
```bash
jupyter notebook test.ipynb
```

---

## **👨‍💻 Pembagian Tugas**

### **Anggota 1: Implementasi Model Dasar & Forward Propagation**  
📌 **Koding:**  
✅ Membuat kelas **FFNN** dalam Python  
✅ Menangani **struktur jaringan** (jumlah layer & neuron per layer)  
✅ Implementasi **fungsi aktivasi** (Linear, ReLU, Sigmoid, Tanh, Softmax, Leaky ReLU, ELU)  
✅ Implementasi **forward propagation**  
✅ Implementasi **fungsi loss** (MSE, Binary Cross-Entropy, Categorical Cross-Entropy)  

📌 **Dokumentasi:**  
✅ Menjelaskan **struktur kelas FFNN** dan atributnya dalam laporan  
✅ Menulis cara kerja **forward propagation** di laporan  

---

### **Anggota 2: Backpropagation & Weight Update**  
📌 **Koding:**  
✅ Implementasi **backward propagation** dengan chain rule  
✅ Menghitung **gradien bobot & bias** tiap layer  
✅ Implementasi **weight update** menggunakan **gradient descent**  
✅ Memastikan model bisa menerima **batch input**  
✅ Implementasi **fungsi save & load model**  
✅ Melatih model FFNN menggunakan **hyperparameter yang sama dengan sklearn MLPClassifier**  

📌 **Dokumentasi:**  
✅ Menjelaskan cara kerja **backward propagation & perhitungan gradien**  
✅ Menulis rumus **gradient descent** di laporan  

---

### **Anggota 3: Pengujian, Analisis, dan Perbandingan dengan sklearn**  
📌 **Koding:**  
✅ Implementasi **inisialisasi bobot** (Zero, Uniform, Normal, Xavier, He)  
✅ Menulis **notebook pengujian** (`.ipynb`)  
✅ Menguji model dengan **dataset MNIST_784**  
✅ Menampilkan **grafik loss training & distribusi bobot**  
✅ Membandingkan hasil prediksi FFNN dan sklearn MLPClassifier  
✅ Menyusun tabel/visualisasi perbandingan hasil model  

📌 **Dokumentasi:**  
✅ Menulis hasil **pengujian & analisis hyperparameter**  
✅ Membahas pengaruh **depth, width, fungsi aktivasi, learning rate** terhadap model  
✅ Menulis kesimpulan tentang kelebihan & kekurangan FFNN dari scratch dibandingkan sklearn  
✅ Menyusun analisis performa FFNN vs sklearn MLPClassifier dalam laporan  

---

## **📄 Deliverables**
- 📌 **Source Code** (`src/`)
- 📌 **Notebook Pengujian** (`test.ipynb`)
- 📌 **Laporan PDF** (`doc/laporan.pdf`)
- 📌 **README.md** (Dokumentasi proyek)

---

## **📚 Referensi**
- [Scikit-Learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Feedforward Neural Network Explanation](https://www.jasonosajima.com/forwardprop)
- [Backpropagation Algorithm](https://www.jasonosajima.com/backprop)
