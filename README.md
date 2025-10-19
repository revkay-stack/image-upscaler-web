# 🖼️ Batch Image Upscaler (Streamlit)

Web app untuk upscale **maksimal 10 gambar** per proses. Mendukung:
- **Real-ESRGAN** (jika tersedia + bobot di folder `weights/`)
- Fallback **Lanczos HQ** bila Real-ESRGAN tidak tersedia
- Pratinjau sebelum/sesudah dan unduh ZIP semua hasil

## 🚀 Cara Deploy (Streamlit Community Cloud)
1. Buat repo GitHub baru dan unggah file proyek ini (`app.py`, `requirements.txt`, `README.md` dan opsional folder `weights/`).
2. Buka https://share.streamlit.io dan **Deploy** dari repo tersebut.
3. Pilih file utama: `app.py`. Setelah sukses, kamu akan mendapat URL seperti:
   `https://<nama-app>-<username>.streamlit.app`

> **Catatan:** Torch/RealESRGAN mungkin tidak tersedia di environment gratis; aplikasi otomatis fallback ke **Lanczos** sehingga tetap bisa dipakai.

## 🚀 Alternatif Deploy (Hugging Face Spaces)
1. Buat Space baru: **Streamlit**.
2. Unggah file proyek ini.
3. Dapatkan URL: `https://huggingface.co/spaces/<username>/<space-name>`

## 🧠 Optional: Menambahkan Bobot Real-ESRGAN
- Buat folder `weights/` dan simpan:
  - `RealESRGAN_x4plus.pth`
  - `RealESRGAN_x2plus.pth`
- Tanpa bobot, aplikasi tetap jalan dengan Lanczos HQ.

## 🧪 Jalankan Lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

---
Lisensi: MIT
