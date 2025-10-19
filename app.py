# app.py
# Streamlit app: Batch Image Upscaler (max 10 files per run)
# Features:
# - Upload up to 10 images
# - Choose scale (2x, 3x, 4x)
# - Try Real-ESRGAN if installed & weights are available, else auto-fallback to high-quality Lanczos
# - Preview before/after, and download all outputs as a ZIP

import io
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageFilter

# Optional: Real-ESRGAN
try:
    from realesrgan import RealESRGAN
    import torch
    HAS_REALESRGAN = True
except Exception:
    HAS_REALESRGAN = False

SUPPORTED_TYPES = ("png", "jpg", "jpeg", "webp", "bmp")
WEIGHTS_DIR = Path("weights")

st.set_page_config(page_title="Batch Image Upscaler", page_icon="🖼️", layout="wide")
st.title("🖼️ Batch Image Upscaler")
st.caption("Upscale hingga **10 gambar sekaligus**. Pilih Real-ESRGAN bila tersedia atau gunakan resize berkualitas tinggi (Lanczos).")

with st.sidebar:
    st.header("Pengaturan")
    method = st.selectbox(
        "Metode upscaling",
        [
            "Auto (Real-ESRGAN jika tersedia)",
            "Real-ESRGAN saja",
            "Lanczos HQ (fallback)",
        ],
        help="Auto akan mencoba Real-ESRGAN; jika tidak tersedia, jatuh ke Lanczos."
    )
    scale = st.select_slider("Skala", options=[2, 3, 4], value=4)
    sharpen = st.slider("Penajaman (fallback)", min_value=0, max_value=3, value=1, help="Hanya untuk metode Lanczos: menambah ketajaman setelah resize.")
    suffix = st.text_input("Akhiran nama file output", value=f"x{scale}")
    st.markdown("---")
    st.markdown("**Status Real-ESRGAN**: ")
    if HAS_REALESRGAN:
        gpu = False
        try:
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        st.success(f"Terpasang ✅ | CUDA: {'Ya' if gpu else 'Tidak'}")
        # Cek bobot
        w2 = (WEIGHTS_DIR / "RealESRGAN_x2plus.pth").exists()
        w4 = (WEIGHTS_DIR / "RealESRGAN_x4plus.pth").exists()
        st.write(f"Bobot x2: {'✅' if w2 else '❌'} | Bobot x4: {'✅' if w4 else '❌'}")
    else:
        st.warning("Real-ESRGAN tidak terpasang. Aplikasi akan menggunakan Lanczos.")

uploaded = st.file_uploader(
    "Pilih hingga 10 gambar",
    type=list(SUPPORTED_TYPES),
    accept_multiple_files=True,
)

# Enforce max 10 files
if uploaded and len(uploaded) > 10:
    st.error("Maksimal 10 gambar per proses. Hapus sebagian lalu coba lagi.")
    uploaded = uploaded[:10]

col_btn, col_clear = st.columns([1,1])
start = col_btn.button("🚀 Proses Upscale")
if col_clear.button("🧹 Bersihkan hasil sesi"):
    st.experimental_rerun()


def upscale_lanczos(img: Image.Image, factor: int, sharpen_steps: int = 1) -> Image.Image:
    new_size = (img.width * factor, img.height * factor)
    out = img.resize(new_size, Image.LANCZOS)
    for _ in range(sharpen_steps):
        out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
    return out


def upscale_realesrgan(img: Image.Image, factor: int) -> Image.Image:
    if not HAS_REALESRGAN:
        raise RuntimeError("RealESRGAN tidak tersedia")
    if factor not in (2, 4):
        chosen = 4
    else:
        chosen = factor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealESRGAN(device, scale=chosen)

    # Tentukan bobot
    weights_path = WEIGHTS_DIR / ("RealESRGAN_x2plus.pth" if chosen == 2 else "RealESRGAN_x4plus.pth")
    if not weights_path.exists():
        raise FileNotFoundError(f"Bobot tidak ditemukan: {weights_path}")

    model.load_weights(str(weights_path))

    # Pastikan mode RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    out = model.predict(img)

    if factor == 3:
        target = (img.width * 3, img.height * 3)
        out = out.resize(target, Image.LANCZOS)
    return out


@st.cache_data(show_spinner=False)
def _bytes_from_image(pil_img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

results = []

if start:
    if not uploaded:
        st.warning("Harap unggah minimal 1 gambar.")
    else:
        progress = st.progress(0)
        status = st.empty()
        total = len(uploaded)

        # Tentukan eksekusi metode
        force_realesrgan = (method == "Real-ESRGAN saja")
        prefer_realesrgan = (method == "Auto (Real-ESRGAN jika tersedia)")

        for idx, f in enumerate(uploaded, start=1):
            status.text(f"Memproses {idx}/{total}: {f.name}")
            try:
                img = Image.open(f)
                img.load()
            except Exception as e:
                st.error(f"Gagal membuka {f.name}: {e}")
                progress.progress(idx/total)
                continue

            use_realesrgan = False
            if force_realesrgan:
                use_realesrgan = True
                if not HAS_REALESRGAN:
                    st.error("Real-ESRGAN tidak tersedia, ubah metode atau instal terlebih dahulu.")
                    progress.progress(idx/total)
                    continue
            elif prefer_realesrgan and HAS_REALESRGAN:
                use_realesrgan = True

            try:
                if use_realesrgan:
                    up = upscale_realesrgan(img, scale)
                else:
                    up = upscale_lanczos(img, scale, sharpen_steps=sharpen)
            except Exception as e:
                st.warning(f"Real-ESRGAN gagal untuk {f.name}: {e}. Menggunakan Lanczos.")
                up = upscale_lanczos(img, scale, sharpen_steps=sharpen)

            # Simpan hasil in-memory & tampilkan
            ext = f.name.split('.')[-1].lower()
            fmt = 'PNG' if ext == 'png' else 'JPEG'
            out_name = f"{Path(f.name).stem}_{suffix}.{ 'png' if fmt=='PNG' else 'jpg'}"

            before_col, after_col = st.columns(2)
            with before_col:
                st.image(img, caption=f"Sebelum: {f.name} ({img.width}×{img.height})", use_container_width=True)
            with after_col:
                st.image(up, caption=f"Sesudah: {out_name} ({up.width}×{up.height})", use_container_width=True)

            data = _bytes_from_image(up, fmt)
            st.download_button(
                label=f"⬇️ Unduh {out_name}",
                data=data,
                file_name=out_name,
                mime=f"image/{'png' if fmt=='PNG' else 'jpeg'}",
            )

            results.append((out_name, data))
            progress.progress(idx/total)

        status.text("Selesai ✅")

if results:
    # Buat ZIP semua hasil
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"upscaled_{ts}.zip"
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, data in results:
            zf.writestr(fname, data)
    st.download_button("📦 Unduh semua sebagai ZIP", data=zip_buf.getvalue(), file_name=zip_name, mime="application/zip")

st.markdown("---")
st.markdown(
    """
    **Tips kualitas**
    - Real-ESRGAN memberi detail lebih baik dibanding resize biasa. Pastikan paket **realesrgan** & bobot model tersedia di folder `weights/`.
    - Gunakan skala 2x untuk foto besar, 4x untuk ikon/ilustrasi kecil.
    - Format keluaran otomatis dipilih: PNG jika sumber PNG, selain itu JPEG.
    """
)
