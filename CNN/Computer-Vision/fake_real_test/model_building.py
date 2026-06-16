import pandas as pd
import numpy as np
import os
import warnings

# Suppress TensorFlow verbose logging and JPEG warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress JPEG decoder warnings (extraneous bytes, premature end, etc)
warnings.filterwarnings('ignore', message='.*extraneous bytes before marker.*')
warnings.filterwarnings('ignore', message='.*premature end of data segment.*')
warnings.filterwarnings('ignore', category=UserWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle


# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

dir = os.path.dirname(os.path.abspath(__file__))

def load_img(path, sub):
    new_path = path
    ds = tf.keras.utils.image_dataset_from_directory(
        directory = new_path,
        labels = "inferred",
        label_mode='int',
        validation_split = 0.2,
        subset = sub,
        batch_size=48,  
        image_size=(128, 128),
        seed = 42
    )

    scale_image = tf.keras.layers.Rescaling(1./ 255)
    scale_df = ds.map(lambda x, y : (scale_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    scale_df = scale_df.apply(tf.data.experimental.ignore_errors())

    """# Add data augmentation ONLY for training dataset
    if sub == 'training':
    # Create augmentation pipeline
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),           # Rotate ±20 degrees
            tf.keras.layers.RandomFlip("horizontal"),      # Flip horizontally
            tf.keras.layers.RandomZoom(0.1),               # Zoom in/out by 20%
            tf.keras.layers.RandomTranslation(0.1, 0.1),   # Shift image by 20%
            tf.keras.layers.RandomBrightness(0.1),         # Brightness ±20%
            tf.keras.layers.RandomContrast(0.1),           # Contrast ±20%
        ])

        scale_df = scale_df.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)"""

    return scale_df.prefetch(buffer_size=tf.data.AUTOTUNE)


print("\nLoading training and validation datasets...")
print("(Note: Any corrupted images will be skipped automatically during training)\n")
train_ds = load_img(r'D:\machine_learning_data\fake_real_data\train', 'training')
val_ds = load_img(r'D:\machine_learning_data\fake_real_data\train', 'validation')


model = tf.keras.models.load_model(os.path.join(dir, 'best_model_checkpoint.keras'))

model.summary()



early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Checkpoint to save best model during training (for resumability)
checkpoint = ModelCheckpoint(
    os.path.join(dir, 'best_model_checkpoint.keras'),
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

print("\nStarting training (batch_size=48, image_size=112x112, mixed_precision enabled)...\n")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,  
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\n✓ Training complete!")
model.save(os.path.join(dir, 'fake_test_modelv4.keras'))
print(f"✓ Model saved to: {os.path.join(dir, 'fake_test_modelv4.keras')}")