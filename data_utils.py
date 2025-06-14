from keras.utils import image_dataset_from_directory
from keras import layers



# Normalization layer (0-255 to 0-1)
normalization_layer = layers.Rescaling(1. / 255)

def normalize_ds(ds):
    """
    Applies normalization (0-255 to 0-1) to a tf.data.Dataset.

    Args:
        ds (tf.data.Dataset): The dataset to normalize.

    Returns:
        tf.data.Dataset: Normalized dataset.
    """
    return ds.map(lambda x, y: (normalization_layer(x), y))


def augment_ds(ds, augmentation_layer):
    """
    Applies a data augmentation pipeline to a dataset.

    Args:
        ds (tf.data.Dataset): The input dataset to augment.
        augmentation_layer (tf.keras.Sequential): A Keras Sequential model containing augmentation layers.

    Returns:
        tf.data.Dataset: The augmented dataset.
    """
    return ds.map(lambda x, y: (augmentation_layer(x, training=True), y))


    
def load_train_val_ds(
    img_size, 
    data_dir, 
    seed, 
    batch_size, 
    validation_split
):

    """
    Loads training and validation datasets from a directory.

    Args:
        img_size (tuple): Target image size, e.g., (96, 96).
        data_dir (str): Path to the train and validation dataset folder, which should contain one subfolder per class (e.g., 'cats/', 'dogs/').
        seed (int): Random seed used to ensure reproducibility when splitting the data.
        batch_size (int): Number of samples per batch.
        validation_split (float): Fraction of the data to use for validation (e.g., 0.3 means 70% train, 30% val).

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: A tuple containing the training and validation datasets.
    """
    
    train_dataset = image_dataset_from_directory(
        directory=data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    val_dataset = image_dataset_from_directory(
        directory=data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    return train_dataset, val_dataset


def load_test_ds(
    img_size,
    data_dir,
    batch_size
):
    """
    Loads a test dataset from a directory in deterministic (non-shuffled) order.

    Args:
        data_dir (str): Path to test dataset directory
        img_size (tuple): Target image size, e.g., (224, 224)
        batch_size (int): Batch size for test loading

    Returns:
        tf.data.Dataset: A batched, normalized test dataset with shuffle disabled
    """
    test_dataset = image_dataset_from_directory(
        directory=data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    return test_dataset

