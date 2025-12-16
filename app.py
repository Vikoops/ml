import io
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2

MODEL_PATH = "emotion_model.keras"

CLASS_NAMES = ["angry", "sad", "happy", "surprise", "neutral"]

IMG_HEIGHT = 96
IMG_WIDTH = 96

HAAR_PATH = "haarcascade_frontalface_default.xml"

st.set_page_config(page_title="Emotion Recognition", layout="centered")

@st.cache_resource
def load_emotion_model(path: str):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_face_cascade(path: str):
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise FileNotFoundError(f"Не удалось загрузить Haar-cascade по пути: {path}")
    return cascade


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def bgr_to_pil(bgr_img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def detect_largest_face(bgr_img: np.ndarray, cascade: cv2.CascadeClassifier):

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None, None
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    face_bgr = bgr_img[y:y+h, x:x+w].copy()
    return (x, y, w, h), face_bgr


def preprocess_bgr_for_model(face_bgr: np.ndarray) -> np.ndarray:
    """BGR лицо -> (1, H, W, 3) float32 [0..1]"""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    x = face_resized.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def predict_from_face(face_bgr: np.ndarray, model: tf.keras.Model):
    x = preprocess_bgr_for_model(face_bgr)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    return pred_label, probs

st.title("Распознавание эмоций по лицу (с детекцией лица)")
st.caption("Пайплайн: изображение → детекция лица (Haar) → обрезка → классификация эмоции (CNN).")

try:
    model = load_emotion_model(MODEL_PATH)
except Exception as e:
    st.error(
        f"Не удалось загрузить модель '{MODEL_PATH}'. Убедись, что файл лежит рядом с app.py.\n\nОшибка: {e}"
    )
    st.stop()

try:
    face_cascade = load_face_cascade(HAAR_PATH)
except Exception as e:
    st.error(f"Не удалось загрузить Haar-cascade.\n\nОшибка: {e}")
    st.stop()

mode = st.radio("Источник изображения", ["Камера", "Загрузка фото"], horizontal=True)

img_pil = None

if mode == "Камера":
    cam = st.camera_input("Сделайте снимок")
    if cam is not None:
        img_pil = Image.open(io.BytesIO(cam.getvalue()))
else:
    uploaded = st.file_uploader("Загрузите фото (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img_pil = Image.open(uploaded)

if img_pil is None:
    st.info("Выберите режим и предоставьте изображение, чтобы получить предсказание.")
    st.stop()

st.subheader("Исходное изображение")
st.image(img_pil, use_container_width=True)

bgr = pil_to_bgr(img_pil)
bbox, face_bgr = detect_largest_face(bgr, face_cascade)

if face_bgr is None:
    st.warning("Лицо не найдено. Выполняется классификация по всему изображению (fallback).")
    face_bgr = bgr.copy()
    bbox = None

pred_label, probs = predict_from_face(face_bgr, model)

st.subheader("Детекция лица")
bgr_vis = bgr.copy()

if bbox is not None:
    x, y, w, h = bbox
    cv2.rectangle(bgr_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    st.image(bgr_to_pil(bgr_vis), caption="Найденное лицо (рамка)", use_container_width=True)
else:
    st.image(bgr_to_pil(bgr_vis), caption="Рамка не построена (лицо не найдено)", use_container_width=True)

st.subheader("Обрезанное лицо, подаваемое в модель")
st.image(bgr_to_pil(face_bgr), use_container_width=True)

st.subheader("Результат")
st.metric("Эмоция", pred_label)

st.write("Вероятности по классам:")
prob_rows = sorted(
    [(CLASS_NAMES[i], float(probs[i])) for i in range(len(CLASS_NAMES))],
    key=lambda x: x[1],
    reverse=True
)
st.table({"class": [r[0] for r in prob_rows], "probability": [f"{r[1]:.3f}" for r in prob_rows]})
