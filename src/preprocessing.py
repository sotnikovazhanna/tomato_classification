import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 16
RANDOM_SEED = 42


def get_project_root():
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root


def load_data_paths(data_path=None):
    if data_path is None:
        project_root = get_project_root()
        data_path = os.path.join(project_root, 'data', 'processed')

    healthy_paths = []
    blight_paths = []

    healthy_dir = os.path.join(data_path, 'healthy')
    blight_dir = os.path.join(data_path, 'late_blight')

    print(f"\n Looking for data in: {data_path}")
    print(f"       Healthy folder: {healthy_dir}")
    print(f"       Late blight folder: {blight_dir}")

    if not os.path.exists(healthy_dir):
        print(f" Folder {healthy_dir} not found")
        print("       Check folder structure:")
        print("       project_root/data/processed/healthy/")
        print("       project_root/data/processed/late_blight/")
        return None, None

    if not os.path.exists(blight_dir):
        print(f" Folder {blight_dir} not found")
        return None, None

    for img in os.listdir(healthy_dir):
        if img.endswith(('.jpg', '.JPG', '.png', '.jpeg')):
            healthy_paths.append(os.path.join(healthy_dir, img))

    for img in os.listdir(blight_dir):
        if img.endswith(('.jpg', '.JPG', '.png', '.jpeg')):
            blight_paths.append(os.path.join(blight_dir, img))

    print(f"\n Images found:")
    print(f"       Healthy: {len(healthy_paths)}")
    print(f"       Late blight: {len(blight_paths)}")

    return healthy_paths, blight_paths


def load_images_from_paths(paths, labels=None, normalize=True):
    images = []
    for i, path in enumerate(paths):
        if i % 500 == 0:
            print(f"       Loaded {i}/{len(paths)} images")
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if normalize:
            img = img.astype(np.float32) / 255.0
        images.append(img)

    X = np.array(images)
    if labels is not None:
        y = np.array(labels[:len(X)])
        return X, y
    return X


def load_data_for_ml():
    print("LOADING DATA FOR ML METHODS")

    healthy_paths, blight_paths = load_data_paths()

    if healthy_paths is None or blight_paths is None:
        return None, None, None, None, None, None

    all_paths = healthy_paths + blight_paths
    all_labels = [0] * len(healthy_paths) + [1] * len(blight_paths)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.3, random_state=RANDOM_SEED, stratify=all_labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_labels
    )

    print(f"\n Dataset sizes:")
    print(f"       Train: {len(train_paths)}")
    print(f"       Val: {len(val_paths)}")
    print(f"       Test: {len(test_paths)}")

    print("\n Loading images into memory")
    print("       Training set:")
    X_train, y_train = load_images_from_paths(train_paths, train_labels, normalize=True)
    print("       Validation set:")
    X_val, y_val = load_images_from_paths(val_paths, val_labels, normalize=True)
    print("       Test set:")
    X_test, y_test = load_images_from_paths(test_paths, test_labels, normalize=True)

    print(f"\n Data loaded:")
    print(f"       X_train: {X_train.shape}")
    print(f"       X_val: {X_val.shape}")
    print(f"       X_test: {X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_generators_from_paths(healthy_paths, blight_paths, batch_size=16):
    all_paths = healthy_paths + blight_paths
    all_labels = [0] * len(healthy_paths) + [1] * len(blight_paths)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.3, random_state=RANDOM_SEED, stratify=all_labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_labels
    )

    print(f"\n Dataset sizes:")
    print(f"       Train: {len(train_paths)}")
    print(f"       Val: {len(val_paths)}")
    print(f"       Test: {len(test_paths)}")

    def image_generator(paths, labels, augment=False):
        datagen = ImageDataGenerator(
            rotation_range=20 if augment else 0,
            width_shift_range=0.1 if augment else 0,
            height_shift_range=0.1 if augment else 0,
            horizontal_flip=True if augment else False,
        )

        def generator():
            for path, label in zip(paths, labels):
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype(np.float32) / 255.0

                if augment:
                    img = datagen.random_transform(img)

                yield img, label

        return generator

    train_dataset = tf.data.Dataset.from_generator(
        image_generator(train_paths, train_labels, augment=True),
        output_types=(tf.float32, tf.int32),
        output_shapes=((IMG_SIZE, IMG_SIZE, 3), ())
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        image_generator(val_paths, val_labels, augment=False),
        output_types=(tf.float32, tf.int32),
        output_shapes=((IMG_SIZE, IMG_SIZE, 3), ())
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_generator(
        image_generator(test_paths, test_labels, augment=False),
        output_types=(tf.float32, tf.int32),
        output_shapes=((IMG_SIZE, IMG_SIZE, 3), ())
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, (test_paths, test_labels)


def load_and_split_data():
    return load_data_for_ml()


if __name__ == "__main__":
    print("DATA LOADING TEST")

    healthy_paths, blight_paths = load_data_paths()

    if healthy_paths and blight_paths:
        print(f"\n Healthy sample: {healthy_paths[0]}")
        print(f" Late blight sample: {blight_paths[0]}")

        result = load_data_for_ml()

        if result[0] is not None:
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = result
            print(f"\n ML data loaded successfully")
            print(f"       X_train shape: {X_train.shape}")
            print(f"       y_train class distribution: {np.bincount(y_train)}")
    else:
        print("\n Data not found. Check:")
        print("1. Folders 'healthy' and 'late_blight' exist in data/processed/")
        print("2. Folders contain .jpg images")