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

# # Predict large number
# def predict(num):
#   result = ""
#   for i in str(num):
#     sample = X[int(i)]
#     sample = sample.reshape(1, -1)
#     sample = scaler.transform(sample)
#     test_predict = model.predict(sample)[0]
#     result += str(test_predict)
#   return result
    
def get_number(image_bytes):
    # convert to numpy (for web)
    # np_arr = np.frombuffer(image_bytes, np.uint8)
    # img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    # For local
    img = cv2.imread(image_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Không đọc được ảnh!")

    img = 255 - img  # swap colors
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # Find all the counters
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Không tìm thấy chữ số!")

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])  # sort
    result = ""
    for (x, y, w, h) in bounding_boxes:
        digit = thresh[y:y+h, x:x+w]
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = digit

        resized = cv2.resize(square, (8, 8), interpolation=cv2.INTER_AREA)
        normalized = (resized / 255.0) * 16
        vector = normalized.flatten().astype(np.float32)
        vector = vector.reshape(1, -1)
        vector = scaler.transform(vector)
        prediction = model.predict(vector)
        result += str(prediction[0])
    return result
