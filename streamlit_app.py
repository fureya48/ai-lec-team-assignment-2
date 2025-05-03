import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model sekali saat startup
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_cifar100.h5")  # Pastikan file model sudah ada
    return model

model = load_model()

# CIFAR-100 label names
label_names = [
    'Apple', 'Aquarium Fish', 'Baby', 'Bear', 'Beaver', 'Bed', 'Bee', 'Beetle', 'Bicycle', 'Bottle',
    'Bowl', 'Boy', 'Bridge', 'Bus', 'Butterfly', 'Camel', 'Can', 'Castle', 'Caterpillar', 'Cattle',
    'Chair', 'Chimpanzee', 'Clock', 'Cloud', 'Cockroach', 'Couch', 'Crab', 'Crocodile', 'Cup', 'Dinosaur',
    'Dolphin', 'Elephant', 'Flatfish', 'Forest', 'Fox', 'Girl', 'Hamster', 'House', 'Kangaroo', 'Computer Keyboard',
    'Lamp', 'Lawn Mower', 'Leopard', 'Lion', 'Lizard', 'Lobster', 'Man', 'Maple Tree', 'Motorcycle', 'Mountain',
    'Mouse', 'Mushroom', 'Oak Tree', 'Orange', 'Orchid', 'Otter', 'Palm Tree', 'Pear', 'Pickup Truck', 'Pine Tree',
    'Plain', 'Plate', 'Poppy', 'Porcupine', 'Possum', 'Rabbit', 'Raccoon', 'Ray', 'Road', 'Rocket',
    'Rose', 'Sea', 'Seal', 'Shark', 'Shrew', 'Skunk', 'Skyscraper', 'Snail', 'Snake', 'Spider',
    'Squirrel', 'Streetcar', 'Sunflower', 'Sweet Pepper', 'Table', 'Tank', 'Telephone', 'Television', 'Tiger', 'Tractor',
    'Train', 'Trout', 'Tulip', 'Turtle', 'Wardrobe', 'Whale', 'Willow Tree', 'Wolf', 'Woman', 'Worm'
]

# Fungsi untuk memproses gambar
def preprocess_image(image, target_size=(32, 32)):
    image = image.resize(target_size)  # Ukuran gambar CIFAR-100 adalah 32x32
    image = np.array(image) / 255.0  # Normalisasi pixel
    image = np.expand_dims(image, axis=0)  # Menambahkan dimensi batch
    return image

# Streamlit UI
st.title("🧠 Image Classifier with CNN")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diupload", use_container_width=True)

    processed_image = preprocess_image(image)

    if st.button("🔍 Prediksi"):
        prediction = model.predict(processed_image)
        
        # Menentukan kelas yang diprediksi
        predicted_class = label_names[np.argmax(prediction)]

        st.success(f"Hasil Prediksi: **{predicted_class}**")
