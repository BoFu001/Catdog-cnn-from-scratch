import os
import pickle
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from typing import Optional, Dict, Tuple

# Loads a saved model and its training history if available.
# If not found, it trains a new model, saves it, and stores the training history as a .pkl file.
# This prevents retraining when the model already exists.
# 
# Parameters:
# - model_dir: Directory path to store or load the model and training history.
# - model_name: filename for the saved model (e.g., .keras)
# - history_name: filename for the saved training history (.pkl)
# - train_data: preprocessed training dataset
# - val_data: preprocessed validation dataset
# - img_size: input image size tuple (e.g., (96, 96))
# - create_model_fn: function to build a new model (e.g. create_model_v1)
# - epochs: number of training epochs to use when training a new model
#
# Returns:
# - model: the trained or loaded Keras model
# - history: training history dictionary, or None if loaded model and history not found


def get_model_and_history(
    model_dir: str,
    model_name: str,
    history_name: str,
    train_data,
    val_data,
    img_size: tuple,
    create_model_fn,
    epochs: int,
    callbacks=None  # Optional callbacks argument
) -> Tuple[Model, Optional[Dict]]:
    
    model_path = os.path.join(model_dir, model_name)
    history_path = os.path.join(model_dir, history_name)


    # Define ModelCheckpoint
    checkpoint_cb = ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )

    if os.path.exists(model_path):
        print("Model already exists, loading it.\n")
        model = load_model(model_path)

        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                history = pickle.load(f)
        else:
            history = None
    else:
        print("Model not found. Training a new model.\n")
        model = create_model_fn(img_size)

        # Combine checkpoint with any provided callbacks
        if callbacks is None:
            callbacks = [checkpoint_cb]
        elif isinstance(callbacks, list):
            callbacks = [checkpoint_cb] + callbacks
        else:
            callbacks = [checkpoint_cb, callbacks]

        
        history_obj = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks)
        history = history_obj.history

        model.save(model_path)
        with open(history_path, "wb") as f:
            pickle.dump(history, f)

    return model, history

# Example usage:
# model_name = "model_v1_img50.keras"
# history_name = "model_v1_img50_history.pkl"
# model_v1_img50, history_1 = get_model_and_history(model_name, history_name, train_dataset_50_norm, val_dataset_50_norm, img_size_50, create_model_v1, epochs)