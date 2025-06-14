import matplotlib.pyplot as plt

def plot_training_history(history, title):
    # Handle both dict and History object
    if hasattr(history, 'history'):
        history = history.history

    epochs = range(1, len(history['accuracy']) + 1)
    
    plt.figure(figsize=(12, 10))
    
    # --- Accuracy Plot ---
    plt.subplot(2, 1, 1)
    line_train, = plt.plot(epochs, history['accuracy'], label='Train Accuracy', marker='o')
    line_val, = plt.plot(epochs, history['val_accuracy'], label='Val Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    # Annotation: Max + Last for Accuracy
    color_train = line_train.get_color()
    color_val = line_val.get_color()

    max_acc_epoch = history['accuracy'].index(max(history['accuracy']))
    plt.text(epochs[max_acc_epoch], history['accuracy'][max_acc_epoch] + 0.01,
             f"{history['accuracy'][max_acc_epoch]:.3f}", ha='center', fontsize=9, color=color_train)

    max_val_acc_epoch = history['val_accuracy'].index(max(history['val_accuracy']))
    plt.text(epochs[max_val_acc_epoch], history['val_accuracy'][max_val_acc_epoch] - 0.015,
             f"{history['val_accuracy'][max_val_acc_epoch]:.3f}", ha='center', fontsize=9, color=color_val)

    # Last point
    plt.text(epochs[-1], history['accuracy'][-1] + 0.01,
             f"{history['accuracy'][-1]:.3f}", ha='center', fontsize=9, color=color_train)
    plt.text(epochs[-1], history['val_accuracy'][-1] - 0.015,
             f"{history['val_accuracy'][-1]:.3f}", ha='center', fontsize=9, color=color_val)

    min_y_accuracy = min(history['accuracy'] + history['val_accuracy'])
    max_y_accuracy = max(history['accuracy'] + history['val_accuracy'])
    plt.ylim(min_y_accuracy - 0.03, max_y_accuracy + 0.03)

    # --- Loss Plot ---
    plt.subplot(2, 1, 2)
    line_train_loss, = plt.plot(epochs, history['loss'], label='Train Loss', marker='o')
    line_val_loss, = plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss')
    plt.legend()

    # Annotation: Min + Last for Loss
    color_train_loss = line_train_loss.get_color()
    color_val_loss = line_val_loss.get_color()

    min_loss_epoch = history['loss'].index(min(history['loss']))
    plt.text(epochs[min_loss_epoch], history['loss'][min_loss_epoch] + 0.01,
             f"{history['loss'][min_loss_epoch]:.3f}", ha='center', fontsize=9, color=color_train_loss)

    min_val_loss_epoch = history['val_loss'].index(min(history['val_loss']))
    plt.text(epochs[min_val_loss_epoch], history['val_loss'][min_val_loss_epoch] - 0.015,
             f"{history['val_loss'][min_val_loss_epoch]:.3f}", ha='center', fontsize=9, color=color_val_loss)

    # Last point
    plt.text(epochs[-1], history['loss'][-1] + 0.01,
             f"{history['loss'][-1]:.3f}", ha='center', fontsize=9, color=color_train_loss)
    plt.text(epochs[-1], history['val_loss'][-1] - 0.015,
             f"{history['val_loss'][-1]:.3f}", ha='center', fontsize=9, color=color_val_loss)

    min_y_loss = min(history['loss'] + history['val_loss'])
    max_y_loss = max(history['loss'] + history['val_loss'])
    plt.ylim(min_y_loss - 0.03, max_y_loss + 0.03)

    plt.tight_layout()
    plt.show()
