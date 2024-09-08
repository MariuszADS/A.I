import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
import cv2
import random


# Funkcje do generowania symboli z losowością
def generate_O():
    image = np.zeros((32, 32), dtype=np.uint8)
    center_x, center_y = random.randint(12, 20), random.randint(12, 20)
    radius = random.randint(4, 8)
    cv2.circle(image, (center_x, center_y), radius, (255), thickness=2)
    return image



def generate_X():
    image = np.zeros((32, 32), dtype=np.uint8)
    start_x1, start_y1 = random.randint(8, 12), random.randint(8, 12)
    end_x1, end_y1 = random.randint(20, 24), random.randint(20, 24)
    start_x2, start_y2 = random.randint(20, 24), random.randint(8, 12)
    end_x2, end_y2 = random.randint(8, 12), random.randint(20, 24)
    cv2.line(image, (start_x1, start_y1), (end_x1, end_y1), (255), thickness=2)
    cv2.line(image, (start_x2, start_y2), (end_x2, end_y2), (255), thickness=2)
    return image


def generate_minus():
    image = np.zeros((32, 32), dtype=np.uint8)
    start_x, start_y = random.randint(8, 12), random.randint(14, 18)
    end_x, end_y = random.randint(20, 24), random.randint(14, 18)
    cv2.line(image, (start_x, start_y), (end_x, end_y), (255), thickness=2)
    return image


def generate_i():
    image = np.zeros((32, 32), dtype=np.uint8)
    start_x, start_y = random.randint(8, 12), random.randint(14, 18)
    end_x, end_y = random.randint(20, 24), random.randint(14, 18)
    cv2.line(image, (start_x, start_y), (end_x, end_y), (255), thickness=2)
    return image

def generate_vLine():
    image = np.zeros((32, 32), dtype=np.uint8)
    start_x, start_y = random.randint(14, 18), random.randint(8, 12)
    end_x, end_y = random.randint(14, 18), random.randint(20, 24)
    cv2.line(image, (start_x, start_y), (end_x, end_y), (255), thickness=2)
    return image


# Generowanie danych treningowych
def generate_data(num_samples):
    data = []
    labels = []

    for _ in range(num_samples):
        label = np.random.choice(["O", "X", "-","I","|"])

        if label == "O":
            image = generate_O()
        elif label == "X":
            image = generate_X()
        elif label == "-":
            image = generate_minus()
        elif label == "I":
            image = generate_i()
        elif label == "|":
            image = generate_vLine()

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
    label_map = {"O": 0, "X": 1, "-": 2,"|": 3}
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