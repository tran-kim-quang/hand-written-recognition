import numpy as np
import cv2
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
# Huấn luyện mô hình SVC từ dữ liệu digits
digits_data = datasets.load_digits() # Load dataset handwritten
y = digits_data.target
X = digits_data.images.reshape((len(digits_data.images), -1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # split data into train and test



# Setup model
X_train, X_test, y_train, y_test = train_test_split(        
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = SVC(C=100, gamma=0.01, kernel='rbf')
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
    vector = vector.reshape(1, -1) 
    vector = scaler.transform(vector)
    # Dự đoán
    prediction = model.predict(vector)
    return prediction
