
import tensorflow as tf

IMG_SIZE   = 224
BATCH_SIZE = 32

def _augment_layer():
    """
    Light augmentation tuned for leaf disease images.
    We avoid aggressive hue/saturation shifts because colour is
    diagnostically important (yellowing, rust spots, etc.).
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),   # max 8% rotation
        tf.keras.layers.RandomZoom(0.1),         # max 10% zoom
        tf.keras.layers.RandomContrast(0.1),     # mild contrast jitter
    ])

def load_datasets(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Loads train and val splits from data_dir/train and data_dir/val.
    Returns (train_ds, val_ds, class_names).
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )
    class_names = train_ds.class_names
    augment    = _augment_layer()
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input  # scales to [-1, 1]

    train_ds = train_ds.map(
        lambda x, y: (preprocess(augment(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (preprocess(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

def load_test_dataset(test_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Loads test split without shuffling, for deterministic evaluation."""
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    test_ds    = test_ds.map(lambda x, y: (preprocess(x), y))
    return test_ds, test_ds.class_names
