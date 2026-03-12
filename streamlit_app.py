import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


u8 = lambda x: np.clip(x, 0, 255).astype(np.uint8)

def rgb_np(pil: Image.Image) -> np.ndarray:
    return np.asarray(pil.convert("RGB"), dtype=np.uint8)

def gray_u8(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def add_noise(g: np.ndarray, mode: str, sigma: float, sp: float) -> np.ndarray:
    if mode == "none":
        return g
    f = g.astype(np.float32)
    if mode == "gaussian":
        return u8(f + np.random.normal(0, sigma, f.shape).astype(np.float32))
    if mode == "salt & pepper":
        out = f.copy()
        n = int(sp * out.size)
        if n <= 0:
            return g
        ys = np.random.randint(0, out.shape[0], n)
        xs = np.random.randint(0, out.shape[1], n)
        h = n // 2
        out[ys[:h], xs[:h]] = 0
        out[ys[h:], xs[h:]] = 255
        return u8(out)
    if mode == "speckle":
        n = np.random.normal(0, sigma / 255.0, f.shape).astype(np.float32)
        return u8(f + f * n)
    return g

def eq(g: np.ndarray, mode: str, clip: float, grid: int) -> np.ndarray:
    if mode == "histeq":
        return cv2.equalizeHist(g)
    if mode == "clahe":
        return cv2.createCLAHE(float(clip), (int(grid), int(grid))).apply(g)
    return g

def win(g: np.ndarray, mode: str, k: int, s: float) -> np.ndarray:
    if mode == "none":
        return g
    k = int(max(1, k))
    k += (k % 2 == 0)
    if mode == "box":
        return cv2.blur(g, (k, k))
    if mode == "gaussian":
        return cv2.GaussianBlur(g, (k, k), float(s))
    return g

def med(g: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k <= 1:
        return g
    k += (k % 2 == 0)
    return cv2.medianBlur(g, k)

def pipeline(rgb: np.ndarray, force_gray: bool, eq_mode: str, win_mode: str, noise_mode: str,
             median_k: int, win_k: int, gauss_sigma: float, noise_sigma: float, sp: float,
             clahe_clip: float, clahe_grid: int) -> np.ndarray:
    g = gray_u8(rgb) if (force_gray or rgb.ndim == 3) else rgb.copy()
    g = u8(g)
    g = eq(g, eq_mode, clahe_clip, clahe_grid)
    g = win(g, win_mode, win_k, gauss_sigma)
    g = add_noise(g, noise_mode, noise_sigma, sp)
    return med(g, median_k)

def fft_mag(g: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(g.astype(np.float32)))
    m = np.log1p(np.abs(f))
    return u8(m / (m.max() + 1e-9) * 255.0)

def show(img: np.ndarray, title: str, gray=True):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.imshow(img, cmap="gray", vmin=0, vmax=255) if gray else ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    st.pyplot(fig, clear_figure=True)


st.session_state.setdefault("img_rgb", None)
st.session_state.setdefault("force_gray", False)
st.session_state.setdefault("show_fft", False)

L, R = st.columns([1, 2], gap="large")

with L:
    up = st.file_uploader(" ", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], label_visibility="collapsed")

    if st.button("Завантажити зображення", use_container_width=True):
        if up is not None:
            st.session_state.img_rgb = rgb_np(Image.open(up))
            st.session_state.show_fft = False

    st.session_state.show_fft = st.button(
        "Фурʼє-аналіз", use_container_width=True,
        disabled=st.session_state.img_rgb is None
    ) or st.session_state.show_fft

    if st.button("Rgb2gray", use_container_width=True, disabled=st.session_state.img_rgb is None):
        st.session_state.force_gray = not st.session_state.force_gray

    if st.button("Очистити все", use_container_width=True):
        st.session_state.img_rgb = None
        st.session_state.force_gray = False
        st.session_state.show_fft = False

with R:
    eq_mode = st.selectbox("Еквалізація", ["histeq", "clahe", "none"], index=0)
    win_mode = st.selectbox("Віконна фільтрація", ["none", "box", "gaussian"], index=0)
    noise_mode = st.selectbox("Додати шум", ["none", "gaussian", "salt & pepper", "speckle"], index=0)
    median_mode = st.selectbox("Медіанна фільтрація", ["none", "3", "5", "7"], index=0)

    clahe_clip, clahe_grid = 2.0, 8
    win_k, gauss_sigma = 5, 1.0
    noise_sigma, sp = 15.0, 0.02

    with st.expander("Додаткові параметри", expanded=False):
        if eq_mode == "clahe":
            clahe_clip = st.slider("CLAHE clipLimit", 1.0, 10.0, 2.0, 0.5)
            clahe_grid = st.slider("CLAHE tileGridSize", 2, 16, 8, 1)
        if win_mode != "none":
            win_k = st.slider("Розмір ядра (непарний)", 3, 31, 5, 2)
            gauss_sigma = st.slider("Gaussian sigma", 0.1, 5.0, 1.0, 0.1) if win_mode == "gaussian" else 1.0
        if noise_mode != "none":
            noise_sigma = st.slider("Noise sigma (Gaussian/Speckle)", 1.0, 80.0, 15.0, 1.0)
            sp = st.slider("Salt&Pepper amount", 0.0, 0.2, 0.02, 0.01)

    median_k = 1 if median_mode == "none" else int(median_mode)

if st.session_state.img_rgb is None:
    st.stop()

rgb = st.session_state.img_rgb
orig = gray_u8(rgb)

proc = pipeline(
    rgb, st.session_state.force_gray,
    eq_mode, win_mode, noise_mode,
    median_k, win_k, gauss_sigma,
    noise_sigma, sp,
    clahe_clip, clahe_grid
)

A, B = st.columns(2, gap="large")
with A:
    st.subheader("Оригінальне зображення")
    show(orig, "Оригінальне зображення", gray=True)

with B:
    if st.session_state.show_fft:
        st.subheader("Фурʼє-аналіз")
        show(fft_mag(proc), "Fourier magnitude (log)", gray=True)
    else:
        title = {"histeq": "Histeq equalization", "clahe": "CLAHE equalization", "none": "Оброблене зображення"}[eq_mode]
        st.subheader(title)
        show(proc, title, gray=True)

st.caption(
    f"Rgb2gray: {'ON' if st.session_state.force_gray else 'OFF'} | "
    f"FFT: {'ON' if st.session_state.show_fft else 'OFF'}"
)