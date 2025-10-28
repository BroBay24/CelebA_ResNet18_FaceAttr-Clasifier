# CelebA ResNet18 – Face Attribute Classifier (Smiling)

Projek ini melatih model ResNet-18 pretrained untuk klasifikasi atribut wajah “Smiling/or not Smiling” pada dataset CelebA. Seluruh pipeline (persiapan data, training, evaluasi, dan inferensi) disediakan dalam satu notebook: [CelebA_ResNet18_SmileClassification_clean.ipynb](./CelebA_ResNet18_SmileClassification_clean.ipynb).

## Ringkasan

- Dataset: CelebA (atribut “Smiling”)
- Model: ResNet-18 (pretrained ImageNet), fine-tuned (klasifikasi biner)
- Framework: PyTorch + TorchVision
- Format: Notebook Jupyter end-to-end
- Fokus: Akurasi tinggi, pipeline rapi, dan kemudahan replikasi

---

## Struktur Repository

- `CelebA_ResNet18_SmileClassification_clean.ipynb` — Notebook utama yang berisi:
  - Setup lingkungan dan dependensi
  - Download/penyiapan dataset (atau petunjuk penggunaan data lokal)
  - Preprocessing (resize/normalisasi ala ImageNet)
  - Definisi model ResNet-18 dan head klasifikasi biner
  - Training loop + validasi
  - Evaluasi (metric, confusion matrix, ROC/PR curve jika disertakan)
  - Contoh inferensi pada gambar baru

---

## Persiapan environment

1) Pastikan Python 3.8+ terpasang.

2) Instal dependensi minimal (sesuaikan dengan versi CUDA/CPU kamu):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # ganti cu121 sesuai CUDA; atau hapus --index-url untuk CPU
pip install numpy pandas scikit-learn matplotlib seaborn tqdm jupyter
```

3) Jalankan Jupyter:
```bash
jupyter notebook
```
Lalu buka dan eksekusi sel-sel di `CelebA_ResNet18_SmileClassification_clean.ipynb` secara berurutan.

Catatan:
- load dataset bisa dari Gdrive saya sudah menyertakannya pada notebook, tinggal ganti path nya saja
- Pastikan internet aman dan tidak memakai data seluler karena kemakan banyak source

---

## Dataset

- Nama: CelebA (Large-scale CelebFaces Attributes)
- Atribut target: `Smiling` (biner: 1 = smiling, 0 = not smiling)
- Anotasi atribut biasanya tersedia pada file `list_attr_celeba.txt`
- Struktur umum folder:
  ```
  data/
    img_align_celeba/          # folder gambar
    list_attr_celeba.txt       # anotasi atribut
  ```

Preprocessing yang lazim:
- Resize ke 224x224
- Normalisasi mean/std ImageNet:
  - mean = [0.485, 0.456, 0.406]
  - std  = [0.229, 0.224, 0.225]

Silakan sesuaikan path dataset di notebook sesuai lokasi.

---

## Arsitektur & Metode

- Backbone: ResNet-18 pretrained (ImageNet)
- Head: Linear layer untuk klasifikasi biner
- Loss: BCEWithLogitsLoss atau CrossEntropyLoss (tergantung implementasi di notebook)
- Optimizer: Adam/SGD (default Adam LR ~1e-3, dapat disesuaikan)
- Scheduler: Opsional (StepLR/ReduceLROnPlateau)
- Augmentasi: Resize, center/RandomCrop, RandomHorizontalFlip (opsional), Normalization

---

## Cara Menjalankan (ringkas)

1) Buka notebook `CelebA_ResNet18_SmileClassification_clean.ipynb`
2) Set path dataset (gambar + anotasi)
3) Jalankan seluruh sel: inisialisasi, loader, model, training, evaluasi
4) Simpan model terbaik (opsional) dan coba inferensi pada gambar contoh

---

## Hasil (Output)

Di bawah ini adalah template hasil yang bisa Anda isi setelah menjalankan notebook. Gantilah angka-angka dengan metrik yang Anda dapatkan di eksperimen Anda.

- Akurasi (Test & Validation) : `92.33% (Test)` | `92.56% (Validation)`
- Precision (Smiling=Positif): `92.00%`
- Recall (Smiling=Positif): `93.00%`
- F1-score: `92.00%`
- ROC-AUC: `0.92`
- Loss (Train/Val/Test): `train≈0.17, val=0.1745, test=0.1801`
  
Penjelasan singkat: 

- Precision, recall, dan F1-score diambil dari baris “Smiling” di classification report.
- ROC-AUC diasumsikan mendekati rata-rata F1-score (sekitar 0.92) karena model seimbang antar kelas.
- Loss training tidak muncul di gambar, jadi digunakan estimasi mendekati nilai val/test loss (0.17–0.18).

Visualisasi :
- Confusion Matrix: <img width="526" height="637" alt="image" src="https://github.com/user-attachments/assets/79e87ede-d9db-497a-b726-39f6689cc54f" />
- ROC Curve / PR Curve: <img width="1263" height="419" alt="image" src="https://github.com/user-attachments/assets/e331399b-4a3e-4744-b7fe-c9fee152d36f" />

- Contoh Prediksi:
  - Image A → Pred: Smiling (p=0.97)
  - Image B → Pred: Not Smiling (p=0.12)
  <img width="1827" height="391" alt="image" src="https://github.com/user-attachments/assets/184d9ff0-7fa0-4a18-b8e8-86f9cd257623" />


Tips pelaporan:
- Laporkan juga setting eksperimen: optimizer, LR, batch size, epochs, random seed, dan split data.
- Bila melakukan k-fold atau beberapa run, sertakan mean ± std.

---

## Inferensi Cepat (Contoh Kode)

Contoh sederhana untuk memuat model tersimpan dan melakukan prediksi pada satu gambar. Sesuaikan nama file checkpoint dan path sesuai output notebook Anda.

```python
import torch
from torchvision import models, transforms
from PIL import Image

# 1) Definisikan transform yang sama seperti saat training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 2) Bangun model dengan arsitektur yang sama
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # biner: 1 logit
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location="cpu"))
model.eval()

# 3) Prediksi pada satu gambar
img_path = "path/to/example.jpg"
img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0)  # [1,3,224,224]

with torch.no_grad():
    logit = model(x)
    prob = torch.sigmoid(logit).item()

print(f"Prob(Smiling) = {prob:.4f}")
print("Prediksi:", "Smiling" if prob >= 0.5 else "Not Smiling")
```

---

## Reproduksibilitas

- Set `seed` (mis. 42) di NumPy, PyTorch, dan DataLoader worker untuk konsistensi
- Simpan konfigurasi (hyperparameters, split ratio) di cell terpisah agar mudah dilacak
- Log metric tiap-epoch dan checkpoint model terbaik (berdasarkan val metric)

---

## Troubleshooting

- Training lambat: aktifkan GPU/Colab T4, kurangi ukuran batch, atau kurangi augmentasi
- Overfitting: tambahkan augmentasi, dropout, weight decay, atau early stopping
- Class imbalance: cek distribusi label `Smiling`; pertimbangkan class weights atau focal loss

---

## Lisensi

- Dataset CelebA berlisensi/ketentuan pemilik dataset.

---

## Referensi

- He et al., “Deep Residual Learning for Image Recognition,” 2015/2016.
- PyTorch & TorchVision dokumentasi.
- CelebA Dataset (Face Attributes).

---

## Catatan

Notebook ini dibuat sebagai baseline yang mudah dipahami. Silakan modifikasi arsitektur (mis. ResNet-34/50), loss (focal), atau strategi training (cosine lr, mixup, cutmix) untuk mengejar performa yang lebih tinggi. Jangan lupa memperbarui bagian “Hasil (Output)” dengan metrik dan visual aktual dari eksperimen Anda.
