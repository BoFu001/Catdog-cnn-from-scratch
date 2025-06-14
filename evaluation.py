import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, test_ds, class_names=None, model_name=None):
    """
    Evaluate the model on test dataset and display precision, F1-score, and confusion matrix.

    Args:
        model (tf.keras.Model): Trained Keras model.
        test_ds (tf.data.Dataset): Normalized test dataset.
        class_names (list, optional): Label names for confusion matrix. Default is ["Class 0", "Class 1"].
        model_name (str, optional): For labeling confusion matrix title.
    """
    y_true, y_pred = [], []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.round(preds).flatten())

    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Precision: {precision:.3f}")
    print(f"F1-score: {f1:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    class_names = class_names or ["Class 0", "Class 1"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    title = f"Confusion Matrix – {model_name}" if model_name else "Confusion Matrix"
    plt.title(title)
    plt.grid(False)
    plt.show()


def show_predictions(model, test_ds, class_names, num_images=12,title="Model Predictions"):
    """
    Display a few predictions from the model on unnormalized test dataset.

    Args:
        model (tf.keras.Model): Trained Keras model.
        test_ds (tf.data.Dataset): Unnormalized test dataset.
        class_names (list): Label names like ['Cat', 'Dog'].
        num_images (int): Number of images to show.
        title (str): Title displayed above the grid.
    """
    for images, labels in test_ds.take(1):
        preds = model.predict(images)
        preds_rounded = np.round(preds).astype(int).flatten()
        plt.figure(figsize=(15, 10))
        plt.suptitle(title, fontsize=16, y=1.02)
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(3, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            true_label = class_names[int(labels[i])]
            pred_label = class_names[int(preds_rounded[i])]
            plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()
