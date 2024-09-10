import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Wczytywanie modelu
model = load_model("symbol_recognition_model.keras")

# Funkcja do rysowania i rozpoznawania symboli
def draw_and_predict():
    # Tworzenie pustego obrazka 512x512
    img = np.zeros((512, 512), dtype=np.uint8)

    # Rysowanie na obrazku
    def draw(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(img, (x, y), 15, (255, 255, 255), -2)

    cv2.namedWindow("Draw")
    cv2.setMouseCallback("Draw", draw)

    while True:
        cv2.imshow("Draw", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == ord('p'):  # 'p' key to predict
            # Przeskalowanie i normalizacja obrazka
            resized_img = cv2.resize(img, (32, 32)).reshape(1, 32, 32, 1) / 255.0
            prediction = model.predict(resized_img)
            label_map = {0: "human",1 : "trapezoid",2 : "pistol", 3 : "kite"}
            predicted_label = label_map[np.argmax(prediction)]
            print(f"Rozpoznany symbol: {predicted_label}")
        elif key == ord('c'):  # 'c' key to clear
            img.fill(0)

    cv2.destroyAllWindows()

# Uruchomienie funkcji rysowania i rozpoznawania
draw_and_predict()