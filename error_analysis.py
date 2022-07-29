from dataset_loader import load_dataset_split
import keras
from keras.metrics import SparseCategoricalAccuracy, Accuracy
import numpy as np
import os
import tensorflow as tf


def evaluate_model(model: keras.Model, dataset: tf.data.Dataset) -> None:
    """
    Evaluates the model performance and creates a report for error analysis
    :param model: a keras.Model instance
    :param dataset: the dataset being tested
    :return:
    """
    # ground_truth shape: (m,)
    # pred_val has shape (m,)
    ground_truth = extract_ground_truth(dataset)
    predictions = get_predictions(model, dataset)

    accuracy = Accuracy()
    accuracy.update_state(ground_truth, predictions)
    print(f"accuracy: {accuracy.result().numpy()}")

    loss_and_metrics = model.evaluate(x=dataset,
                                      return_dict=True,
                                      verbose=1)
    print(loss_and_metrics)





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
    for element in val_dataset.as_numpy_iterator():
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
    config = {
       "image_size": (256, 256),
        "batch_size": 60
    }
    model = keras.models.load_model("experiments\\test1_data_001\\best.hdf5")
    val_dataset = load_dataset_split("val", "data\\data_001\\val", config, shuffle=False)
    evaluate_model(model, val_dataset)
