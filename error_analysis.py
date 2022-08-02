from CONSTANTS import CLASSES
from dataset_loader import load_dataset_split
import keras
import numpy as np
import os
from sklearn.metrics import classification_report
from tabulate import tabulate
import tensorflow as tf
import util


def evaluate_and_report(config: dict) -> None:
    """
    Evaluates a model performance in the val split of a dataset and creates 2 reports: one at a class-level and
    the other at an aggregate level
    :param config: the configuration file of the experiment, specifies the necessary details to proceed
    :return:
    """

    # Get the model and dataset
    experiment_folder = util.config_get_experiment_dir(config)
    model = keras.models.load_model(os.path.join(experiment_folder,"best.hdf5"))
    dataset = load_dataset_split("val", config, shuffle=False)

    # ground_truth shape: (m,)
    # pred_val has shape (m,)
    ground_truth = extract_ground_truth(dataset)
    predictions = get_predictions(model, dataset)

    # Generate and print aggregated performance across classes
    target_names = CLASSES.values()
    clf_report = classification_report(ground_truth, predictions, target_names=target_names)
    with open(os.path.join(experiment_folder, "classification_report.txt"), "w") as file:
        file.write(str(clf_report))
    print(f"classification report created")

    # Compute most frequent mistakes per-class
    mistakes = create_mistakes_frequencies(ground_truth, predictions)

    # Print mistakes report as a .txt and with image examples
    print_mistakes_report_with_table(mistakes, experiment_folder)
    print(f"most frequent mistakes report created")


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


def create_mistakes_frequencies(ground_truth: np.ndarray, predictions: np.ndarray) -> list:

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








