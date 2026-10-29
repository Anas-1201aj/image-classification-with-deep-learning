
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Configuration
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = Path("dataset/train")
VAL_DIR = Path("dataset/val")


# Load images and labels
def load_dataset(folder):
    images = []
    labels = []

    for path in folder.iterdir():
        if not path.is_file():
            continue

        filename = path.name.lower()

        if filename.startswith("cat"):
            label = 0
        elif filename.startswith("dog"):
            label = 1
        else:
            continue

        try:
            image = tf.keras.utils.load_img(
                path,
                target_size=IMAGE_SIZE
            )
            image = tf.keras.utils.img_to_array(image)

            images.append(image)
            labels.append(label)

        except Exception as error:
            print(f"Skipping {path.name}: {error}")

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.float32)
    )


print("Loading images...")

X_train, y_train = load_dataset(TRAIN_DIR)
X_val, y_val = load_dataset(VAL_DIR)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))

if len(X_train) == 0 or len(X_val) == 0:
    raise ValueError("No images found. Check your dataset folders.")

# Data augmentation
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# Build transfer learning model
inputs = layers.Input(shape=(160, 160, 3))

x = augmentation(inputs)

# MobileNetV2 expects pixel values between -1 and 1
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = models.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save the best model and stop if validation stops improving
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "dog_cat_model.keras",
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# Evaluate best saved model
best_model = tf.keras.models.load_model(
    "dog_cat_model.keras"
)

train_loss, train_accuracy = best_model.evaluate(
    X_train, y_train, verbose=0
)

val_loss, val_accuracy = best_model.evaluate(
    X_val, y_val, verbose=0
)

print("\nTraining complete!")
print(f"Training accuracy: {train_accuracy:.2%}")
print(f"Validation accuracy: {val_accuracy:.2%}")
print("Best model saved as dog_cat_model.keras")
