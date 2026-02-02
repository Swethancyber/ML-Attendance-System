# Image training - Fixed version

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image loader
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    " # enter the location of your file",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Load MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Custom classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

# Final model
model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    train_data,
    epochs=10
)

# Save model
model.save("actors_face_model.h5")
