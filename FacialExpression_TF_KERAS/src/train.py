import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1./255,)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)


emote_model = Sequential()
emote_model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu', input_shape=(48, 48, 1)))
emote_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emote_model.add(MaxPooling2D(2, 2))  # downsampling
emote_model.add(Dropout(0.25))  # prevent overfitting,train more on noise
emote_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emote_model.add(MaxPooling2D(2, 2))
emote_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emote_model.add(MaxPooling2D(2, 2))
emote_model.add(Dropout(0.25))
emote_model.add(Flatten())
emote_model.add(Dense(1024, activation='relu'))  # hidden layer
emote_model.add(Dropout(0.5))
emote_model.add(Dense(7, activation='softmax'))


emote_model.compile(optimizer=Adam(learning_rate=0.0001, decay=1e-6),
                    loss='categorical_crossentropy', metrics=['accuracy'])

# emote_model.summary()
#plot_model(emote_model, show_layer_names=True)
# Found 28709 images belonging to 7 classes. -train
# Found 7178 images belonging to 7 classes. - test

emote_model_info = emote_model.fit_generator(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=7178 // 64
)

# Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.

emote_model.save_weights('model.h5')
