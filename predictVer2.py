import numpy as np
import cv2
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Huấn luyện mô hình SVC từ dữ liệu digits
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)
model = SVC(gamma=0.001)
model.fit(X_train, y_train)

def get_number(image_bytes):
    # Chuyển ảnh byte thành mảng numpy
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Không đọc được ảnh!")

    img = 255 - img  # Đảo màu nếu nền trắng chữ đen
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # Tìm contour lớn nhất
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Không tìm thấy chữ số!")

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    digit = thresh[y:y+h, x:x+w]

    # Cân bằng và căn giữa thành ảnh vuông
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = digit

    # Resize về 8x8 và chuẩn hóa giá trị pixel
    resized = cv2.resize(square, (8, 8), interpolation=cv2.INTER_AREA)
    normalized = (resized / 255.0) * 16
    vector = normalized.flatten().astype(np.float32)

    # Dự đoán
    prediction = model.predict([vector])
    return prediction