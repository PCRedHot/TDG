import os

import seaborn as sn
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


from model import get_uncompiled_model, SEED


# tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

train_data_dir = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), 'data', 'training')
test_data_dir = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), 'data', 'testing')

img_height, img_width = 60, 60
batch_size = 32
n_epochs = 100

NUM_WORKERS = 5
MAX_QUEUE_SIZE = 20

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=360,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2)  # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    keep_aspect_ratio=True,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    seed=SEED,
    subset='training')  # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,  # same directory as training data
    target_size=(img_height, img_width),
    keep_aspect_ratio=True,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    seed=SEED,
    subset='validation')  # set as validation data

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,  # same directory as training data
    target_size=(img_height, img_width),
    keep_aspect_ratio=True,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical')

n_classes = np.max(train_generator.classes) + 1

# metrics = ['accuracy', 'val_accuracy']
metrics = ['accuracy']

checkpoint_folder = 'training_4'
checkpoint_folder_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), checkpoint_folder)

if not os.path.exists(checkpoint_folder_path):
    os.makedirs(checkpoint_folder_path)

checkpoint_filepath = os.path.join(
    checkpoint_folder_path, "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5")
checkpoint_best_filepath = os.path.join(
    checkpoint_folder_path, "weights-best.hdf5")

plot_model(get_uncompiled_model(n_classes, (img_width, img_height)),
           to_file=os.path.join(checkpoint_folder_path, 'model.png'), show_shapes=True, show_layer_names=True)

train = False
test = False
batch_test = True

if train:
    model = get_uncompiled_model(n_classes, (img_width, img_height))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=metrics)

    csv_logger = CSVLogger(os.path.join(
        checkpoint_folder_path, "model_history_log.csv"), append=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True)
    cp_best_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_best_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=n_epochs,
        workers=NUM_WORKERS,
        max_queue_size=MAX_QUEUE_SIZE,
        callbacks=[csv_logger, cp_callback, cp_best_callback, es])

if test:
    if batch_test:
        for f in reversed(os.listdir(checkpoint_folder_path)):
            if not f.endswith('.hdf5'): continue
            if f.endswith('best.hdf5'): continue

            model = get_uncompiled_model(n_classes, (img_width, img_height))
            model.load_weights(os.path.join(checkpoint_folder_path, f))
            model.compile(optimizer='adam', loss='categorical_crossentropy',
                    metrics=metrics)

            results = model.evaluate(
                test_generator, workers=NUM_WORKERS, max_queue_size=MAX_QUEUE_SIZE,
            )
            print(f"{f} test loss, test acc:", results)

            prediction = model.predict(
                test_generator, workers=NUM_WORKERS,  max_queue_size=MAX_QUEUE_SIZE,)
            prediction = np.argmax(prediction, axis=1)

            matrix = confusion_matrix(test_generator.labels, prediction)
            classes_sort = sorted(os.listdir(test_data_dir))
            df_cm = pd.DataFrame(matrix, index=[i for i in classes_sort],
                                columns=[i for i in classes_sort])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True)
            # plt.show()

            plt.savefig(os.path.join(checkpoint_folder_path,
                        f'result_confusion_matrix_{f}.png'))


    model = get_uncompiled_model(n_classes, (img_width, img_height))
    model.load_weights(checkpoint_best_filepath)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics=metrics)

    results = model.evaluate(
        test_generator, workers=NUM_WORKERS, max_queue_size=MAX_QUEUE_SIZE,
    )
    print("test loss, test acc:", results)
    
    prediction = model.predict(
        test_generator, workers=NUM_WORKERS,  max_queue_size=MAX_QUEUE_SIZE,)
    prediction = np.argmax(prediction, axis=1)

    matrix = confusion_matrix(test_generator.labels, prediction)
    classes_sort = sorted(os.listdir(test_data_dir))
    df_cm = pd.DataFrame(matrix, index=[i for i in classes_sort],
                        columns=[i for i in classes_sort])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    # plt.show()

    plt.savefig(os.path.join(checkpoint_folder_path,
                'result_confusion_matrix_best.png'))
