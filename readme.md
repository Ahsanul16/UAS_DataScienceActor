# Project: Actor Gender Classification based on Names

Proyek Machine Learning ini bertujuan untuk mengklasifikasikan gender aktor ('M' atau 'F') berdasarkan nama panggung mereka. Proyek ini membandingkan performa tiga pendekatan model yang berbeda: Baseline, Machine Learning Tradisional, dan Deep Learning.
link laporan dan video penjelasan:https://drive.google.com/drive/u/0/folders/17M1U7vUR2cCV5w5c7FxgHh9_pfJOmg-w

## 📂 Struktur Folder
project/
│
├── data/                   # Dataset (actors.html) - Upload manual
├── notebooks/              # Jupyter Notebooks (Google Colab)
├── src/                    # Source code tambahan
├── models/                 # Model tersimpan (.pkl & .h5)
│   ├── model_baseline.pkl  # Dummy Classifier
│   ├── model_rf.pkl        # Random Forest + TF-IDF
│   └── model_cnn.h5        # CNN 1D
│
├── images/                 # Hasil Visualisasi EDA
│   └── r/
│       ├── 1_distribusi_gender.png
│       ├── 2_top_roles.png
│       └── 3_panjang_nama.png
│
├── requirements.txt        # Daftar library python
└── README.md               # Dokumentasi Proyek



## 📊 Dataset
* **Sumber:** `actors.html` (Gio's Movie files).
* **Metode Parsing:** Manual text splitting (karena format HTML legacy).
* **Fitur Utama:**
    * `name`: Nama panggung aktor (Input).
    * `gender`: Gender aktor (Target: 'M' atau 'F').
    * `role`: Tipe peran (untuk analisis EDA).
* **Preprocessing:**
    * Filtering data valid (hanya M/F).
    * Encoding label (F=0, M=1).
    * Tokenization (Character-level) untuk Deep Learning.
    * TF-IDF (Character N-gram) untuk Random Forest.

## 🧠 Model & Evaluasi
Kami melatih dan mengevaluasi 3 model dengan metrik **Akurasi**:

| Model | Deskripsi | Input Features |
| :--- | :--- | :--- |
| **1. Baseline** | Dummy Classifier (Most Frequent) | N/A |
| **2. Random Forest** | Ensemble Learning (100 Trees) | TF-IDF (Char 2-3 gram) |
| **3. CNN 1D** | Deep Learning (Keras/TensorFlow) | Char Embedding Sequence |

*Hasil evaluasi detail dapat dilihat pada log output notebook.*

## 📈 Visualisasi (EDA)
Terdapat 3 visualisasi utama yang tersimpan di folder `images/r/`:
1.  **Distribusi Gender:** Melihat keseimbangan kelas dataset.
2.  **Top Roles:** Menampilkan 10 tipe peran yang paling sering muncul.
3.  **Panjang Nama:** Histogram distribusi jumlah karakter pada nama.

## 🚀 Cara Menjalankan (Google Colab)
1.  Upload folder proyek atau file notebook ke Google Drive/Colab.
2.  Pastikan file `actors.html` diupload ke dalam folder `project/data/`.
3.  Jalankan semua sel secara berurutan.
4.  Model akan otomatis tersimpan di folder `models/`.

## 🛠 Dependencies
* Python 3.x
* Pandas & NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib & Seaborn
* BeautifulSoup4
""
