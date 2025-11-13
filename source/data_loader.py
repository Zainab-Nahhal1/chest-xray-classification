
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def create_generators(data_dir, target_size=(150,150), batch_size=32, validation_split=0.2):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=validation_split
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=target_size, batch_size=batch_size,
        color_mode='rgb', class_mode='binary', subset='training', shuffle=True
    )
    validation_generator = train_datagen.flow_from_directory(
        train_dir, target_size=target_size, batch_size=batch_size,
        color_mode='rgb', class_mode='binary', subset='validation', shuffle=True
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=target_size, batch_size=batch_size,
        color_mode='rgb', class_mode='binary', shuffle=False
    )

    class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    class_weights = dict(enumerate(class_weights))

    return train_generator, validation_generator, test_generator, class_weights
