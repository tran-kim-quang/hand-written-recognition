from sklearn import datasets
# import matplotlib.pyplot as plt
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import cv2
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.model_selection import GridSearchCV, KFold
# from sklearn import metrics
# import seaborn as sns
# Load data
digits_data = datasets.load_digits() # Load dataset handwritten
y = digits_data.target
X = digits_data.images.reshape((len(digits_data.images), -1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # split data into train and test

# Parameter
C = 10
gamma = 0.001
kernel = 'rbf'

# Setup model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = SVC(C=C, gamma=gamma, kernel=kernel)
model.fit(X_train, y_train)

def predict_hand_written(img_path):
    # Load img
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # np_arr = np.frombuffer(img_path, np.uint8)
    # img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Không đọc được ảnh!")

    # Binary picture
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find largest counter
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Không tìm thấy chữ số trong ảnh!")

    # Crop bounding box
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    digit = thresh[y:y+h, x:x+w]

    # Make border
    height, width = digit.shape
    padding = abs(height - width) // 2
    if height > width:
        digit = cv2.copyMakeBorder(digit, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=0)
    else:
        digit = cv2.copyMakeBorder(digit, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # Resize to 8x8
    resized = cv2.resize(digit, (8, 8), interpolation=cv2.INTER_AREA)

    # Normalize into 0-16
    # (:Attribute Information: 8x8 image of integer pixels in the range 0..16.)
    normalized = (resized / 255.0) * 16
    vector = normalized.flatten().astype(np.float32)

    return model.predict([vector])