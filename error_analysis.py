from CONSTANTS import CLASSES
from dataset_loader import load_dataset_split
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report
from tabulate import tabulate
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
import util


def evaluate_and_report(config: dict) -> None:
    """
    Evaluates a model performance in the val split of a dataset and creates 3 reports:
        - Report per class
        - Report aggregated on the classes
        - Visual report showing the top most frequent mistakes with grad Cam
    :param config: the configuration file of the experiment, specifies the necessary details to proceed
    :return:
    """

    # Get the model and dataset
    experiment_folder = util.config_get_experiment_dir(config)
    model = keras.models.load_model(os.path.join(experiment_folder,"best.hdf5"))
    dataset = load_dataset_split("val", config, shuffle=False)

    # ground_truth shape: (m,)
    # pred_val has shape (m,)
    images, ground_truth = extract_images_gt(dataset)
    probabilities, predictions = get_probabilities_and_predictions(model, images, config["batch_size"])

    # Generate and print aggregated performance across classes
    target_names = CLASSES.values()
    clf_report = classification_report(ground_truth, predictions, target_names=target_names)
    with open(os.path.join(experiment_folder, "classification_report.txt"), "w") as file:
        file.write(str(clf_report))
    print(f"classification report created")

    # Compute most frequent mistakes per-class and print the report
    mistakes = create_mistakes_list(ground_truth, predictions)
    print_mistakes_report_with_table(mistakes, experiment_folder, topK=10)
    print(f"most frequent mistakes report created")

    # Now assemble and print the visual report
    create_visual_report(mistakes, images, ground_truth, predictions, probabilities, model, experiment_folder, topK=3)


def print_mistakes_report_with_table(mistakes: list, exp_folder_path: str, topK: int = 10) -> None:
    """
    Table has headers: ["Ranking", "Ground Truth", "Prediction", "Frequency", "% in total predictions", "% in total mistakes"]
    :param mistakes:
    :param topK:
    :return:
    """
    table = []
    for i in range(topK):
        row = mistakes[i]
        ranking = row[0]
        gt_class_index = row[1]
        pred_class_index = row[2]
        frequency = row[3]
        gt_class = CLASSES[gt_class_index]
        pred_class = CLASSES[pred_class_index]
        percentage_within_predictions = row[4]
        percentage_within_mistakes = row[5]
        new_row = [ranking, gt_class, pred_class, frequency, percentage_within_predictions, percentage_within_mistakes]
        table.append(new_row)
    report = tabulate(table, headers=["Ranking",
                                      "Ground Truth",
                                      "Prediction",
                                      "Frequency",
                                      "% in total predictions",
                                      "% in total mistakes"])
    with open(os.path.join(exp_folder_path, "mistakes_report.txt"), "w") as file:
        file.write(str(report))


def create_mistakes_list(ground_truth: np.ndarray, predictions: np.ndarray) -> list:
    """
    Creates a list of 3 tuples, such as (2, 4, 5, 10, 0.11, 0.82) each element representing:
        - Ranking of this frequency mistake (2 or second)
        - ground truth index class (4)
        - predicted index class (5)
        - frequency (10)
        - % within total predictions (0.11)
        - % within total mistakes (0.82)
    :param ground_truth: (m,) numpy array
    :param predictions: (m,) numpy array
    :return:
    """

    # Both gt and pred must be the same shape
    assert ground_truth.shape == predictions.shape, "ground truth and predictions must have similar shape"

    # Joint gt_pred array
    joint_gt_pred = np.column_stack((ground_truth, predictions))

    # Remove correct predictions
    joint_gt_pred_mistakes = joint_gt_pred[joint_gt_pred[:,0] != joint_gt_pred[:,1]]

    # Count the cases
    unique, counts = np.unique(joint_gt_pred_mistakes, return_counts=True, axis=0)
    total_mistakes = sum(counts)
    total_predictions = len(ground_truth)

    # Transform to list and sort in a descending fashion
    gt_pred_count = list(np.column_stack((unique, counts)))
    gt_pred_count.sort(key=lambda x:-x[-1])

    # Add the percentages values
    mistake_list = []
    for i, (gt, pred, count) in enumerate(gt_pred_count):
        percentage_within_predictions = round(count / total_predictions,2)
        percentage_within_mistakes = round(count / total_mistakes, 2)
        mistake_list.append((i+1, gt, pred, count, percentage_within_predictions, percentage_within_mistakes))
    return mistake_list


def get_probabilities_and_predictions(model: keras.Model, images: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Gets the probabilities and the predictions (with argmax) of the model with a given dataset with K classes
    :param model: keras Model
    :param images: of shape (m, h, w, c)
    :param batch_size: batch size to process the images
    :return:
        - probabilities: np array with shape (m, K)
        - predictions: np array with shape (m,)
    """

    # This is a (m,K) array, where m is the number of examples and K is the number of classes
    probabilities = model.predict(x=images, batch_size=batch_size)
    # This is a (m,) vector
    predictions = np.argmax(probabilities, axis=1)
    return probabilities, predictions


def extract_images_gt(dataset: tf.data.Dataset) -> np.ndarray:
    """
    From a tf.data.Dataset extracts the images and the gt
    :param dataset:
    :return:
    """

    ground_truth = None
    images = None
    for img, gt in dataset.as_numpy_iterator():
        if images is None:
            images = img
            ground_truth = gt
        else:
            images = np.concatenate((images, img))
            ground_truth = np.concatenate((ground_truth, gt))
    return images, ground_truth


def create_visual_report(mistakes: list, images: np.ndarray, ground_truth: np.ndarray, predictions: np.ndarray,
                         probabilities: np.ndarray, model: keras.Model, save_directory:str, topK: int = 10):

    for i, mistake in enumerate(mistakes):

        if i == topK:
            break

        ranking, gt_index_class, pred_index_class, frequency, percentage_within_predictions, percentage_within_mistakes = mistake

        # Get all the indices of these mistakes happening
        indices = np.arange(len(ground_truth))
        joint_index_gt_pred = np.column_stack((indices, ground_truth, predictions))

        # Keep only rows where gt and pred belong to the current mistake gt and pred
        joint_index_gt_pred = joint_index_gt_pred[joint_index_gt_pred[:, 1] == gt_index_class]  # filter by gt
        joint_index_gt_pred = joint_index_gt_pred[joint_index_gt_pred[:, 2] == pred_index_class]  # filter by pred

        # Create the visual report for this mistake
        create_visual_report_single_page(model, ranking, gt_index_class, pred_index_class, percentage_within_mistakes,
                                         images, probabilities, joint_index_gt_pred, save_directory)


def create_visual_report_single_page(model: keras.Model, ranking: int, gt_index: int, pred_index: int,
                                     percentage_within_mistakes, images: np.ndarray, probabilities: np.ndarray,
                                     joint_index_gt_pred: np.ndarray, save_directory:str, size:int = 5):

    # Shuffle the rows to pick a random subset of mistakes to show in a single page
    util.shuffle_2D_array(joint_index_gt_pred)

    # Select a subset of the elements
    final_size = min(size, len(joint_index_gt_pred))
    joint_index_gt_pred = joint_index_gt_pred[0:final_size]

    # Assemble the visual report figure:
    # First column: the raw image
    # Second column: Grad CAM
    # Third column: Probabilities
    gt_class = CLASSES[gt_index]
    pred_class = CLASSES[pred_index]
    fig, axs = plt.subplots(final_size, 3, figsize=(14, 14), constrained_layout=True)#
    title = f"Examples of ground truth = {gt_class} and prediction = {pred_class}" \
            f"\nRanking: {ranking} with fraction of mistakes: {percentage_within_mistakes}" \
            f"\nOn the left: input image     On the right: Grad CAM of the prediction"
    fig.suptitle(title, fontsize=10)

    # Assemble the plots of each example
    explainer = GradCAM()
    classes_indices = np.arange(len(CLASSES))
    values = []
    for key, value in CLASSES.items():
        values.append(str(key) + "_" + value)
    x_axis = np.linspace(0, 1.0, 5, endpoint=True)
    for i in range(final_size):
        index = joint_index_gt_pred[i, 0]
        image_i = images[index]
        probabilities_i = probabilities[index]

        # Original image
        axs[i, 0].imshow(image_i.astype("uint8"))
        axs[i, 0].set_axis_off()

        # Grad Cam
        data = ([image_i], None)
        grid = explainer.explain(data, model, class_index=pred_index, image_weight=0.2)
        axs[i, 1].imshow(grid.astype("uint8"))
        axs[i, 1].set_axis_off()

        # Probabilities
        x = CLASSES.values()
        barlist = axs[i, 2].barh(y=classes_indices, width=probabilities_i)
        barlist[gt_index].set_color("green")  # This is the ground truth set to green
        axs[i, 2].set_yticks(classes_indices, labels=values, fontsize=7)
        axs[i, 2].set_xticks(ticks=x_axis, fontsize=7)
        axs[i, 2].invert_yaxis()  # labels read top-to-bottom
        axs[i, 2].set_xlabel('Probability', fontsize=7)

    filename = "error_analysis_top_" + str(ranking).zfill(2)
    plt.savefig(os.path.join(save_directory, filename))
    print(f"Saved visual report: {filename}")


if __name__ == "__main__":
    config_train = {
        "model_name": "test",
        "dataset": "data_001",
        "is_new_experiment": True,
        "image_size": (480, 640),  # height x width
        "batch_size": 16,
        "epochs": 10
    }
    #evaluate_and_report(config_train)









