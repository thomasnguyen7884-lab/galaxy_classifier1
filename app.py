import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Import lại các hàm logic từ file predict.py của bạn
from predict import load_classes, load_model_any, load_image_for_model, is_end2end_image_model, get_embedder

# --- CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(
    page_title="Galaxy Classifier Deep Learning Project",
    page_icon="🌌",
    layout="centered"
)

st.title("🌌 Galaxy Morphology Classifier")
st.write("This interactive web app uses Deep Learning to classify galaxy shapes. Upload an image to see the model's prediction.")

# --- TỰ ĐỘNG LOAD MODEL & CLASSES ---
@st.cache_resource # Giúp lưu vào bộ nhớ đệm, tránh việc load lại model mỗi lần bấm nút
def init_project():
    # Mặc định tìm các file trong thư mục dự án của bạn
    model_path = "galaxy_classifier.h5" 
    classes_path = "class_names.npy"
    
    # Fallback kiểm tra nếu dùng đuôi .keras
    if not os.path.exists(model_path) and os.path.exists("galaxy_classifier.keras"):
        model_path = "galaxy_classifier.keras"
        
    try:
        model = load_model_any(model_path)
        classes = load_classes(classes_path)
        return model, classes, None
    except Exception as e:
        return None, None, str(e)

model, classes, error_msg = init_project()

# Hiển thị thông báo nếu thiếu file model hoặc lớp
if error_msg:
    st.error(f"❌ Error loading model assets: {error_msg}")
    st.info("Please make sure 'galaxy_classifier.h5' (or .keras) and 'class_names.npy' are in the same folder.")
    st.stop()

# --- BỐ CỤC CHỨC NĂNG TẢI ẢNH ---
st.subheader("1. Upload Galaxy Image")
uploaded_file = st.file_uploader("Choose a galaxy image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh người dùng tải lên
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
    with col2:
        st.write("⚙️ **Processing Status:** Image uploaded successfully. Ready to analyze.")
        run_prediction = st.button("🚀 Classify Galaxy Shape", type="primary")

    # --- KHI BẤM NÚT DỰ ĐOÁN ---
    if run_prediction:
        with st.spinner("Analyzing morphology... Please wait..."):
            try:
                # Lưu tạm ảnh để hàm load_image_for_model xử lý giống cấu trúc cũ của bạn
                temp_path = "temp_uploaded_img.png"
                image.save(temp_path)
                
                # Gọi hàm tiền xử lý từ predict.py
                img, x = load_image_for_model(temp_path)
                
                # Dự đoán logic dựa trên loại model (End-to-End hay Embedding)
                if is_end2end_image_model(model):
                    prob = model.predict(x, verbose=0)[0]
                else:
                    embedder = get_embedder()
                    emb = embedder.predict(x, verbose=0)
                    prob = model.predict(emb, verbose=0)[0]
                
                # Xóa file ảnh tạm sau khi xong
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Tính toán kết quả
                pred_idx = int(np.argmax(prob))
                pred_class = classes[pred_idx]
                confidence = prob[pred_idx] * 100
                
                # --- HIỂN THỊ KẾT QUẢ TƯƠNG TÁC ---
                st.subheader("2. Classification Results")
                st.success(f"🏆 **Prediction:** **{pred_class}** ({confidence:.1f}% Confidence)")
                
                # Vẽ biểu đồ cột xác suất bằng Matplotlib chuẩn Streamlit
                fig, ax = plt.subplots(figsize=(6, 3))
                y_pos = np.arange(len(classes))
                
                # Đổ màu nổi bật cho cột có xác suất cao nhất
                colors = ['#1f77b4'] * len(classes)
                colors[pred_idx] = '#2ca02c' # Màu xanh lá cho kết quả dự đoán trúng
                
                ax.barh(y_pos, prob, align='center', color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(classes)
                ax.invert_yaxis()  # Đọc từ trên xuống dưới
                ax.set_xlabel('Probability Score')
                ax.set_title('Model Confidence Breakdown')
                plt.tight_layout()
                
                # Đưa biểu đồ lên web thay vì dùng plt.show()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
