import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập cấu hình trang
st.set_page_config(page_title="Stock Recognition System", page_icon="🍎", layout="wide")

# Tùy chỉnh CSS để làm đẹp giao diện
st.markdown("""
    <style>
    .main {
        background: #1a2526; /* Nền đen cố định */
        padding: 20px;
        transition: all 0.3s ease;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        color: white;
        padding: 10px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content h1 {
        color: #ecf0f1;
        font-family: 'Poppins', sans-serif;
        font-size: 28px;
        text-align: center;
    }
    .sidebar .sidebar-content .stSelectbox label {
        color: #ecf0f1;
        font-family: 'Poppins', sans-serif;
    }
    .sidebar .sidebar-content .stSelectbox div {
        background-color: #2c3e50;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .sidebar .sidebar-content .stSelectbox div:hover {
        background-color: #e74c3c;
    }
    .title {
        font-size: 60px;
        color: #ffffff;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .subtitle {
        font-size: 24px;
        color: #ffffff;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        opacity: 0.9;
    }
    .prediction {
        font-size: 36px;
        color: #27ae60;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
    }
    .confidence {
        font-size: 20px;
        color: #2980b9;
        text-align: center;
        font-family: 'Poppins', sans-serif;
    }
    .card {
        background: #2c3e50; /* Nền thẻ tối để tương phản với chữ trắng */
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    .card-title {
        font-size: 24px;
        color: #ff6b6b; /* Đỏ sáng để nổi bật trên nền tối */
        font-family: 'Poppins', sans-serif;
    }
    .card-content {
        font-size: 16px;
        color: #ffffff; /* Màu trắng để tăng độ tương phản */
        font-family: 'Poppins', sans-serif;
    }
    .loading {
        font-size: 20px;
        color: #e74c3c;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .history-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        font-family: 'Poppins', sans-serif;
        color: #ecf0f1;
    }
    </style>
""", unsafe_allow_html=True)

# Load mô hình đã huấn luyện và biên dịch lại
model = tf.keras.models.load_model('trained_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Danh sách các lớp (36 lớp từ dataset của bạn)
class_names =[
    '7up', 'Aquafina', 'Cocacola', 'Fanta cam', 'Milk', 'Milo', 'Pepsi','apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalapeno',
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas' ,'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
    'turnip', 'watermelon'
]
assert len(class_names) == model.output_shape[1], f"Expected {model.output_shape[1]} class names, but got {len(class_names)}"

# Khởi tạo session state để lưu lịch sử dự đoán
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Hàm tiền xử lý ảnh
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize về kích thước mô hình yêu cầu
    image = np.array(image) / 255.0  # Chuẩn hóa về [0, 1]
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch: (1, 64, 64, 3)
    return image

# Hàm dự đoán và trả về top 5 xác suất
def predict_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_classes =[class_names[i] if i < len(class_names) else "Unknown" for i in top_5_indices]
    top_5_probs = predictions[0][top_5_indices]
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]
    predicted_class = class_names[class_index]
    return predicted_class, confidence, top_5_classes, top_5_probs

# Hàm tạo biểu đồ xác suất
def plot_probabilities(top_5_classes, top_5_probs):
    plt.figure(figsize=(8, 4))
    sns.barplot(x=top_5_probs, y=top_5_classes, palette="Blues_d")
    plt.xlabel("Probability")
    plt.ylabel("Class")
    plt.title("Top 5 Predictions")
    st.pyplot(plt)

# Hàm tải xuống ảnh kết quả
def get_image_download_link(image, predicted_class, confidence):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="prediction_{predicted_class}.png">Download Result</a>'
    return href

# Sidebar
with st.sidebar:
    st.markdown('<h1>Dashboard</h1>', unsafe_allow_html=True)
    app_mode = st.selectbox("Select Page", ["Home", "About Project", "Predict Images"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("- [GitHub Repository](https://github.com/your-repo)")
    st.markdown("- [Dataset Source](https://www.kaggle.com/datasets)")
    st.markdown("---")
    st.markdown("### Contact")
    st.markdown("📧 Email: your.email@example.com")
    st.markdown("📞 Phone: +123-456-7890")

# Main Page
if app_mode == "Home":
    st.markdown('<h1 class="title">🍎 Stock Recognition System 🍌</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A cutting-edge AI system to recognize fruits and vegetables with unparalleled accuracy!</p>', unsafe_allow_html=True)

    # Thêm ảnh nền
    if os.path.exists('home_image.jpeg'):
        st.image('home_image.jpeg', use_container_width=True)
    else:
        st.warning("Home image not found. Please add 'home_image.jpeg' to the project folder.")

    # Thêm thông tin giới thiệu
    st.markdown("---")
    st.markdown("### Why Stock Recognition System?")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    - **Fast and Accurate**: Recognize 36 types of fruits and vegetables with 94% accuracy.<br>
    - **User-Friendly**: Capture images using your webcam or upload from your device.<br>
    - **Powered by AI**: Built with TensorFlow and Streamlit for a seamless experience.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# About Project
elif app_mode == "About Project":
    st.markdown('<h1 class="title">About the Project</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover the dataset and technology behind this innovative system</p>', unsafe_allow_html=True)

    # About Dataset
    st.markdown("### About Dataset")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Dataset Overview</p>', unsafe_allow_html=True)
    st.markdown("""
    <p class="card-content">
    This dataset contains images of 36 different fruits and vegetables, collected for training a deep learning model. The dataset is divided into training, validation, and test sets.
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Hiển thị danh sách Fruits và Vegetables dưới dạng hai cột
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">Fruits 🍎</p>', unsafe_allow_html=True)
        fruits = "Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango"
        st.markdown(f'<p class="card-content">{fruits}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">Vegetables 🥕</p>', unsafe_allow_html=True)
        vegetables = ("Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, "
                      "Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, "
                      "Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant")
        st.markdown(f'<p class="card-content">{vegetables}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Thêm thông tin về công nghệ
    st.markdown("### Technology Stack")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Tools & Libraries</p>', unsafe_allow_html=True)
    st.markdown("""
    <p class="card-content">
    - **TensorFlow**: For building and training the CNN model.<br>
    - **Streamlit**: For creating this interactive web application.<br>
    - **Python**: Core programming language.<br>
    - **Matplotlib & Seaborn**: For visualizing prediction probabilities.
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Predict Images
elif app_mode == "Predict Images":
    st.markdown('<h1 class="title">Predict Images</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Capture or upload an image to identify fruits and vegetables!</p>', unsafe_allow_html=True)

    # Tạo hai cột: một cho ảnh, một cho kết quả
    col1, col2 = st.columns(2)

    # Cột 1: Webcam và upload ảnh
    with col1:
        st.markdown("### 📸 Capture or Upload Image")
        # Webcam
        captured_image = st.camera_input("Take a picture with your webcam")
        # Upload ảnh
        uploaded_image = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

        # Xử lý ảnh
        image = None
        if captured_image is not None:
            image = Image.open(captured_image)
            st.image(image, caption="Captured Image", use_container_width=True)
        elif uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    # Cột 2: Hiển thị kết quả
    with col2:
        st.markdown("### 🔍 Prediction Result")
        if image is not None:
            # Hiệu ứng loading
            with st.spinner('Predicting...'):
                predicted_class, confidence, top_5_classes, top_5_probs = predict_image(image)

            # Hiển thị kết quả
            st.markdown(f'<p class="prediction">It\'s a {predicted_class}!</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="confidence">Confidence: {confidence:.2%}</p>', unsafe_allow_html=True)

            # Lưu vào lịch sử dự đoán
            st.session_state.prediction_history.append({
                "image": image.copy(),
                "class": predicted_class,
                "confidence": confidence
            })

            # Hiển thị biểu đồ xác suất
            st.markdown("#### Top 5 Predictions")
            plot_probabilities(top_5_classes, top_5_probs)

            # Nút tải xuống kết quả
            st.markdown("#### Download Result")
            st.markdown(get_image_download_link(image, predicted_class, confidence), unsafe_allow_html=True)

            # Nút chia sẻ (giả lập)
            if st.button("Share Result"):
                st.success(f"Shared: Predicted {predicted_class} with {confidence:.2%} confidence!")

        else:
            st.markdown('<p class="subtitle">Waiting for an image...</p>', unsafe_allow_html=True)

    # Hiển thị lịch sử dự đoán
    st.markdown("---")
    st.markdown("### Prediction History")
    if st.session_state.prediction_history:
        for i, history in enumerate(st.session_state.prediction_history):
            st.markdown('<div class="history-item">', unsafe_allow_html=True)
            st.image(history["image"], width=100)
            st.markdown(f"**{history['class']}** (Confidence: {history['confidence']:.2%})")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="subtitle">No predictions yet.</p>', unsafe_allow_html=True)

# Thêm footer
st.markdown("---")
st.markdown('<p class="subtitle">Built with ❤️ by [Your Name] | Powered by Streamlit & TensorFlow</p>', unsafe_allow_html=True)