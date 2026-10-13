# predict.py  — ưu tiên Keras 3 (.keras), fallback tf.keras (.h5)
import argparse, os, numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf  # vẫn cần cho tf.keras & preprocess

# ===== Ưu tiên Keras 3, fallback sang tf.keras =====
try:
    import keras as k3
    from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    KERAS3 = True
    print(">>> Đang dùng Keras 3 (standalone) để load .keras")
except Exception:
    from tensorflow import keras as tfk
    from tensorflow.keras.applications import MobileNetV2, preprocess_input
    k3 = None
    KERAS3 = False
    print(">>> Không có Keras 3 — dùng tf.keras")

IMG_SIZE = (224, 224)

def _resolve_path(p: str) -> str:
    """Tìm file theo nhiều cách: chính nó, join với CWD, join với thư mục script,
    và quét theo basename trong project. Trả về đường dẫn tìm thấy đầu tiên."""
    cands = []

    # 1) như người dùng đưa vào
    cands.append(p)

    # 2) join với CWD
    cands.append(os.path.join(os.getcwd(), p))

    # 3) join với thư mục chứa predict.py
    base = os.path.dirname(os.path.abspath(__file__))
    cands.append(os.path.join(base, p))

    # chuẩn hoá
    cands = [os.path.normpath(x) for x in cands]

    for c in cands:
        if os.path.exists(c):
            return c

    # 4) quét theo basename trong cả project (thư mục của predict.py)
    name = os.path.basename(p)
    for root, _, files in os.walk(base):
        if name in files:
            return os.path.join(root, name)

    raise FileNotFoundError("Không tìm thấy: {}\nĐã thử:\n  {}".format(p, "\n  ".join(cands)))

def load_classes(path="class_names.npy"):
    path = _resolve_path(path)
    return np.load(path, allow_pickle=True).tolist()

def load_model_any(path="galaxy_classifier.h5"):
    """Ưu tiên dùng Keras 3 để load .keras; còn lại dùng tf.keras."""
    path = _resolve_path(path)
    # Nếu có Keras 3 và file là .keras -> dùng Keras 3 (deserialize chuẩn)
    if KERAS3 and path.lower().endswith(".keras"):
        return k3.models.load_model(path, compile=False)

    # Fallback: dùng tf.keras (hỗ trợ .h5, SavedModel, và 1 số .keras đơn giản)
    if KERAS3:
        # có Keras3 vẫn có thể load .h5 bằng tf.keras
        from tensorflow import keras as tfk
        return tfk.models.load_model(path, compile=False)
    else:
        return tfk.models.load_model(path, compile=False)

def get_embedder():
    """Trích đặc trưng bằng MobileNetV2 pretrained ImageNet (pooling='avg' -> 1280-d)."""
    base = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        pooling="avg",
        weights="imagenet"
    )
    return base

def load_image_for_model(img_path):
    img_path = _resolve_path(img_path)
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    x = np.array(img).astype("float32")[None, ...]
    x = preprocess_input(x)
    return img, x  # img để hiển thị, x để đưa vào mạng

def is_end2end_image_model(model):
    """Phân biệt model nhận ảnh trực tiếp (HWC) hay nhận embedding (1280-d)."""
    try:
        shp = model.input_shape
    except Exception:
        shp = getattr(model.layers[0], "input_shape", None)
    if shp is None:
        return True  # fallback

    if isinstance(shp, list):
        shp = shp[0]

    # (None, 224, 224, 3)
    if isinstance(shp, tuple) and len(shp) == 4 and shp[1:4] == (224, 224, 3):
        return True
    # (None, 1280) hoặc vài kích cỡ phổ biến khác
    if isinstance(shp, tuple) and len(shp) == 2 and shp[1] in (1280, 1024, 2048):
        return False
    return True

def predict_image(model, classes, img_path):
    img, x = load_image_for_model(img_path)

    if is_end2end_image_model(model):
        prob = model.predict(x, verbose=0)[0]
    else:
        embedder = get_embedder()
        emb = embedder.predict(x, verbose=0)
        prob = model.predict(emb, verbose=0)[0]

    pred = classes[int(np.argmax(prob))]
    print("Prediction:", pred)
    print("Probs:", dict(zip(classes, np.round(prob, 3))))

    plt.imshow(img); plt.axis("off"); plt.title(f"Pred: {pred}")
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="galaxy_classifier.h5",
                    help="Đường dẫn model (.h5 hoặc .keras)")
    ap.add_argument("--classes", default="class_names.npy",
                    help="File numpy chứa tên lớp")
    ap.add_argument("--image", required=True, help="Ảnh cần dự đoán")
    args = ap.parse_args()

    classes = load_classes(args.classes)
    model = load_model_any(args.model)
    predict_image(model, classes, args.image)

if __name__ == "__main__":
    main()
