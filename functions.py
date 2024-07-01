import os
import pandas as pd
import numpy as np
import io
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image

def train_vgg16_model(data_path, filenames, labels):
    df = pd.DataFrame({'filename': filenames, 'label': [str(label) for label in labels]})

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=data_path,
        x_col='filename',
        y_col='label',
        target_size=(250, 250),
        class_mode='binary',
        batch_size=32
    )

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(250, 250, 3))
    base_model.trainable = False

    inputs = Input(shape=(250, 250, 3))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10
    )

    model.save('model.keras')


def predict_image(model_path, image_bytes):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please train the model first.")

    model = load_model(model_path)

    img = image.load_img(io.BytesIO(image_bytes), target_size=(250, 250))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "melanoma" if prediction[0][0] > 0.5 else "nevus"

    return result
