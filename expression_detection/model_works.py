from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Rescaling
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow as tf
import numpy as np
import os

def make_model(trainImageFolder:str=None, epochs:int=16, saveModelName:str=None, inputData=None, inputLabels=None):
    """
        A function for training an image recognition model. Outputs a trained model.

        Optionally can be saved to a local file rather than having to re-train the model each time the function is called. Will return it if such a model is found.

        If trainImageFolder is passed, it will use the built-in Tensorflow utility for quick loading and automatic labeling. Preferred method over the one found in this project.
        If it's not passed, then you must pass inputData and inputLabels from the preprocess.load_data() function.
    """
    if trainImageFolder is not None:
        train_data = image_dataset_from_directory(
            trainImageFolder,
            validation_split=0.0,
            image_size=(48,48),
        )
        classNames = train_data.class_names
        num_classes = len(train_data.class_names)

    if os.path.exists(saveModelName):
        print("Loading model...")
        return load_model(saveModelName), classNames

    if trainImageFolder is not None:
        train_data = image_dataset_from_directory(
            trainImageFolder,
            validation_split=0.0,
            image_size=(48,48),
        )
        classNames = train_data.class_names
        num_classes = len(train_data.class_names)
    
    else:
        train_data = inputData
        classNames = None
        num_classes = np.unique(inputLabels)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    model = Sequential([
        Input(shape=(48, 48,3)),
        Rescaling(1./255),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax', name='outputs')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    print("Training model...")
    model.fit(train_data, epochs=epochs)

    if saveModelName:
        print(f"Saving model to {saveModelName}...")
        model.save(saveModelName)

    return model, classNames

def validate_model(model, validateImagePath:str=None, validateData=None, validateLabels=None):
    """
        A function that returns the total test loss and accuracy for a given model and validation set.

        If validateImagePath is passed, then it will use the built-in Tensorflow method for dataset building. Preferred over the alternative.
        If not, you will have to pass validateData and validateLabels acquired from preprocess.load_data().
    """
    if validateImagePath is not None:
        if os.path.isdir(validateImagePath):
            validation_data = image_dataset_from_directory(
                validateImagePath,
                validation_split=0,
                image_size=(48,48)
            )

            test_loss, test_acc = model.evaluate(validation_data, verbose=2)
            return (test_loss, test_acc)

    else:
        test_loss, test_acc = model.evaluate(validateData, validateLabels, verbose=2)
        return (test_loss, test_acc)

def predict_image(model, class_names, img_array=None, imagePath:str=None):
    """
        Returns the predicted class and confidence level for a given static image using imagePath, or a frame using img_array.

        If img_array is passed, it should be a NumPy array from a PIL Image.
    """
    if imagePath is not None:
        image = tf.keras.utils.load_img(validateImagePath, target_size=(48, 48))
        img_array = tf.keras.utils.img_to_array(image)

    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    return class_names[np.argmax(predictions)], 100*np.max(predictions)
