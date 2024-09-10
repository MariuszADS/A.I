import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
import cv2
import random


# Funkcje do generowania symboli z losowością

def generate_trapezoid():
    image = np.zeros((32, 32), dtype=np.uint8)

    # Definiujemy losowe współrzędne dla wierzchołków trapezu
    # Górna podstawa (krótsza)
    top_left_x = random.randint(10, 14)
    top_right_x = random.randint(18, 22)
    top_y = random.randint(5, 10)

    # Dolna podstawa (dłuższa)
    bottom_left_x = random.randint(5, 8)
    bottom_right_x = random.randint(24, 26)
    bottom_y = random.randint(24, 28)

    # Wierzchołki trapezu
    points = np.array([
        [top_left_x, top_y],  # Górny lewy
        [top_right_x, top_y],  # Górny prawy
        [bottom_right_x, bottom_y],  # Dolny prawy
        [bottom_left_x, bottom_y]  # Dolny lewy
    ], np.int32)

    # Przekształcamy punkty do kształtu akceptowanego przez OpenCV
    points = points.reshape((-1, 1, 2))

    # Rysujemy trapez na obrazie
    cv2.polylines(image, [points], isClosed=True, color=(255), thickness=1)

    return image


def generate_human():
    image = np.zeros((32, 32), dtype=np.uint8)  # Tworzymy czarny obraz 32x32 pikseli

    # Głowa (kółko)
    center = (16, 8)
    radius = 3
    cv2.circle(image, center, radius, (255), thickness=1)

    # Ciało (linia pionowa)
    top_body = (16, 11)
    bottom_body = (16, 22)
    cv2.line(image, top_body, bottom_body, (255), thickness=1)

    # Ręce (linia pozioma)
    left_hand = (10, 15)
    right_hand = (22, 15)
    cv2.line(image, left_hand, right_hand, (255), thickness=1)

    # Nogi (linie ukośne)
    left_leg_top = (16, 22)
    left_leg_bottom = (12, 30)
    right_leg_top = (16, 22)
    right_leg_bottom = (20, 30)
    cv2.line(image, left_leg_top, left_leg_bottom, (255), thickness=1)
    cv2.line(image, right_leg_top, right_leg_bottom, (255), thickness=1)

    return image

def generate_pistol():
    image = np.zeros((32, 32), dtype=np.uint8)  # Tworzymy czarny obraz 32x32 pikseli

    # Korpus pistoletu (prostokąt)
    body_top_left = (8, 10)
    body_bottom_right = (24, 16)
    cv2.rectangle(image, body_top_left, body_bottom_right, (255), thickness=1)

    # Lufa pistoletu (mały prostokąt)
    barrel_top_left = (24, 12)
    barrel_bottom_right = (30, 14)
    cv2.rectangle(image, barrel_top_left, barrel_bottom_right, (255), thickness=1)

    # Uchwyt pistoletu (ukośna linia)
    grip_top = (10, 16)
    grip_bottom = (6, 26)
    cv2.line(image, grip_top, grip_bottom, (255), thickness=1)

    return image

def generate_kite():
    image = np.zeros((32, 32), dtype=np.uint8)  # Tworzymy czarny obraz 32x32 pikseli

    # Wierzchołki latawca (romb)
    top_point = (16, 6)     # Górny punkt
    right_point = (24, 16)  # Prawy punkt
    bottom_point = (16, 26) # Dolny punkt
    left_point = (8, 16)    # Lewy punkt

    # Rysowanie boków latawca (romb)
    cv2.line(image, top_point, right_point, (255), thickness=1)
    cv2.line(image, right_point, bottom_point, (255), thickness=1)
    cv2.line(image, bottom_point, left_point, (255), thickness=1)
    cv2.line(image, left_point, top_point, (255), thickness=1)

    # Rysowanie sznurka (linia)
    cv2.line(image, bottom_point, (16, 31), (255), thickness=1)

    return image


# Generowanie danych treningowych
def generate_data(num_samples):
    data = []
    labels = []

    for _ in range(num_samples):
        label = np.random.choice(["trapezoid",'human',"pistol",'kite'])

        if label == "trapezoid":
            image = generate_trapezoid()
        elif label == "human":
            image = generate_human()
        elif label == "pistol":
            image = generate_pistol()
        elif label == "kite":
            image = generate_kite()




        data.append(image)
        labels.append(label)

        # Powiększanie obrazu 8x8 razy
        enlarged_image = cv2.resize(image, (456, 456), interpolation=cv2.INTER_NEAREST)

        # Wyświetlanie wygenerowanego symbolu
        cv2.imshow("Generated Symbol", enlarged_image)
        cv2.waitKey(500)  # Czekaj 500 ms

    data = np.array(data).reshape(-1, 32, 32, 1)
    labels = np.array(labels)

    # Konwersja etykiet na wartości numeryczne
    label_map = {"human" : 0,"trapezoid": 1,"pistol": 2, "kite": 3}
    labels = np.vectorize(label_map.get)(labels)

    return data, labels


# Generowanie danych
data, labels = generate_data(4000)
cv2.destroyAllWindows()

# Normalizacja danych
data = data / 255.0

# Budowanie modelu
model = models.Sequential([
    Input(shape=(32, 32, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
model.fit(data, labels, epochs=15, validation_split=0.2)

# Zapisanie modelu
model.save("symbol_recognition_model.keras")

model.summary()