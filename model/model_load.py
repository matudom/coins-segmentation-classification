import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pathlib

# Načtení modelu bez kompilace
model = load_model('coin_classifier_model.keras', compile=False)

# Kompilace modelu po načtení
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

# Načtení tříd ze souboru
class_names = np.load('class_names.npy')
print(f"Nalezené třídy: {class_names}")


test_dir = input("Vyberte složku s testovacími daty: \n")
test_dir = pathlib.Path(test_dir)

# Kontrola, zda složka existuje
if not test_dir.exists():
    print(f"Chyba: Složka {test_dir} neexistuje.")
    exit(1)

# Funkce pro předpověď třídy obrázku
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Vytvoření batchu

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_class, confidence

# Procházení všech obrázků v testovací složce a předpověď jejich tříd
image_paths = list(test_dir.glob('**/*.jpg'))
if not image_paths:
    print("Chyba: Ve složce nebyly nalezeny žádné obrázky s příponou .jpg.")
    exit(1)

for img_path in image_paths:
    predicted_class, confidence = predict_image(img_path)
    print(f"Obrázek: {img_path} je předpovězen jako {predicted_class} s důvěrou {confidence:.2f}%")