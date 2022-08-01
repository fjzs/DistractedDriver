from CONSTANTS import CLASSES
from dataset_loader import load_dataset_split
import keras
import numpy as np
import os
from sklearn.metrics import classification_report
from tabulate import tabulate
import tensorflow as tf


def evaluate_model(model: keras.Model, dataset_dir: str, split:str, exp_folder_path: str, image_size=(256,256)) -> None:
    """
    Evaluates the model performance and creates a report for error analysis
    :param model: a keras.Model instance
    :param dataset_dir: the dataset directory being tested
    :return:
    """

    # Assemble the dataset
    config = {"batch_size":32, "image_size":image_size}
    dataset = load_dataset_split(split, dataset_dir, config, shuffle=False)

    # ground_truth shape: (m,)
    # pred_val has shape (m,)
    ground_truth = extract_ground_truth(dataset)
    predictions = get_predictions(model, dataset)

    # Generate and print aggregated performance across classes
    target_names = CLASSES.values()
    clf_report = classification_report(ground_truth, predictions, target_names=target_names)
    with open(os.path.join(exp_folder_path, "classification_report.txt"), "w") as file:
        file.write(str(clf_report))

    # Compute most frequent mistakes per-class
    mistakes = create_mistakes_frequencies(ground_truth, predictions, exp_folder_path)

    # Print mistakes report as a .txt and with image examples
    print_mistakes_report_with_table(mistakes, exp_folder_path)


def print_mistakes_report_with_table(mistakes: list, exp_folder_path: str, topK: int = 10) -> None:
    """
    Table has headers: ["Ground Truth", "Prediction", "Frequency"]
    :param mistakes:
    :param topK:
    :return:
    """
    table = []
    for i in range(topK):
        row = mistakes[i]
        gt_class_index = row[0]
        pred_class_index = row[1]
        frequency = row[2]
        gt_class = CLASSES[gt_class_index]
        pred_class = CLASSES[pred_class_index]
        new_row = [gt_class, pred_class, frequency]
        table.append(new_row)
    report = tabulate(table, headers=["Ground Truth", "Prediction", "Frequency"])
    with open(os.path.join(exp_folder_path, "mistakes_report.txt"), "w") as file:
        file.write(str(report))



def create_mistakes_frequencies(ground_truth: np.ndarray, predictions: np.ndarray, exp_folder_path: str) -> list:

    # Both gt and pred must be the same shape
    assert ground_truth.shape == predictions.shape, "ground truth and predictions must have similar shape"

    # Joint gt_pred array
    joint_gt_pred = np.column_stack((ground_truth, predictions))

    # Remove correct predictions
    joint_gt_pred_mistakes = joint_gt_pred[joint_gt_pred[:,0] != joint_gt_pred[:,1]]

    # Count the cases
    unique, counts = np.unique(joint_gt_pred_mistakes, return_counts=True, axis=0)

    # Transform to list and sort in a descending fashion
    gt_pred_count = list(np.column_stack((unique, counts)))
    gt_pred_count.sort(key=lambda x:-x[-1])
    return gt_pred_count


def get_predictions(model: keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    """
    Gets the prediction of the model with a given dataset
    :param model:
    :param dataset: of shape (m, h, w, c)
    :return: predictions, np array with shape (m,)
    """

    # This is a (m,K) array, where m is the number of examples and K is the number of classes
    predictions = model.predict(x=dataset)
    # This is a (m,) vector
    predictions = np.argmax(predictions, axis=1)
    return predictions


def extract_ground_truth(dataset: tf.data.Dataset) -> np.ndarray:
    """
    From a tf.data.Dataset target a numpy 1D vector is retrieved
    :param dataset:
    :return:
    """

    gt_list_of_batches = []
    size = 0
    for element in dataset.as_numpy_iterator():
        target = element[1]
        size += len(target)
        gt_list_of_batches.append(target)

    # Now assemble the numpy array
    gt_nd_array = np.ones(shape=size, dtype=int)*-1
    index = 0
    for batch in gt_list_of_batches:
        batch_size = len(batch)
        gt_nd_array[index: index + batch_size] = batch
        index += batch_size
    return gt_nd_array


if __name__ == "__main__":
    pass








