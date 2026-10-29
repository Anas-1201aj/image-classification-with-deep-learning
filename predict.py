
import tensorflow as tf
import numpy as np
from pathlib import Path

# Load the saved model
model = tf.keras.models.load_model("dog_cat_model.keras")

# Ask the user for an image path
image_path = input("Enter the image path: ").strip().strip('"')

if not Path(image_path).is_file():
    print("Image file not found. Please check the path.")
else:
    # Load and resize the image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(128, 128)
    )

    # Convert image to a numerical array
    image_array = tf.keras.utils.img_to_array(image)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array, verbose=0)[0][0]

    if prediction >= 0.5:
        print(f"Prediction: Dog ({prediction:.2%} model score)")
    else:
        print(f"Prediction: Cat ({1 - prediction:.2%} model score)")
