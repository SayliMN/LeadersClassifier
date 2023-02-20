import numpy as np
import cv2
import joblib
import json
import base64
import os
from wavelet import waveTrans


__class_name_to_number = {}
__class_number_to_name = {}

__model = None


def classify_image(image_base64_data, file_path=None):
    imgs = get_a_cropped_img_if_both_eyes(file_path, image_base64_data)
    result = []

    for img in imgs:
        scaled_img = cv2.resize(img, (32, 32))  # Resized image
        haar_img = waveTrans(img, 'db1', 5)  # Convert it into wavelet transform
        scaled_img_haar = cv2.resize(haar_img, (32, 32))  # Resize the wavelet transform image
        new_img = np.vstack((scaled_img.reshape(32 * 32 * 3, 1), scaled_img_haar.reshape(32 * 32, 1)))  # Vertical stacking of a scaled and a wavelet scaled image

        len_img_arr = 32 * 32 * 3 + 32 * 32

        # predict expects 2D array (in here multiple images), that's why the final array has been reshaped to 1D
        final = new_img.reshape(1, len_img_arr).astype(float)
        result.append(__model.predict(final)[0])
    return result


def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./Artifacts/labels_dict.json", 'r') as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open("./Artifacts/saved_best_model1.pkl", 'rb') as f:
            __model = joblib.load(f)

    print("Loading saved artifacts...done")


# Converting a base64 string into opencv image
def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_a_cropped_img_if_both_eyes(image_path, image_base64_data):
    path_to_haar_clf = "/Opencv/haarcascades/"
    face_clf = cv2.CascadeClassifier(os.getcwd() + path_to_haar_clf + "haarcascade_frontalface_default.xml")
    eye_clf = cv2.CascadeClassifier(os.getcwd() + path_to_haar_clf + "haarcascade_eye.xml")

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_loc = face_clf.detectMultiScale(g_img, 1.2, 3)

    cropped_faces = []
    for (x, y, w, h) in faces_loc:
        c_face = img[y:y+h, x:x+h]
        g_face = g_img[y:y+h, x:x+h]
        eyes_loc = eye_clf.detectMultiScale(g_face)
        if len(eyes_loc) >= 2:
            cropped_faces.append(c_face)
        return cropped_faces


# to get test img as a string
def get_base64_test_image_for_anand():
    with open("b64.txt") as f:
        return f.read()


if __name__ == '__main__':
    load_saved_artifacts()

    print(classify_image(get_base64_test_image_for_anand(), None))
