import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt

# Load CIFAR-100 dataset
(train_images, train_labels), (val_images, val_labels) = cifar100.load_data()

# Rescale pixel values to [0, 1] range
train_images, val_images = train_images / 255.0, val_images / 255.0

# One-hot encode labels
train_labels = tf.keras.utils.to_categorical(train_labels, 100)
val_labels = tf.keras.utils.to_categorical(val_labels, 100)

# Model builder with transfer learning and hyperparameter tuning
def build_model(hp):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),  # CIFAR-100 images are 32x32
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())

    # Batch Normalization
    model.add(layers.BatchNormalization())

    # Dense layer with tuning
    model.add(layers.Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu'))

    # Dropout
    model.add(layers.Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1)))

    # Output layer
    model.add(layers.Dense(100, activation='softmax'))  # 100 classes in CIFAR-100

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Hyperband tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=4,
    factor=3,
    directory='tuner_dir',
    project_name='cifar100_transfer_learning_tuning'
)

# Search for best model
tuner.search(train_images, train_labels, validation_data=(val_images, val_labels), epochs=5)

# Save best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save('best_model.keras')
