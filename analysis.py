from CONSTANTS import CLASSES
from dataset_loader import load_dataset_split
import keras
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report
from tabulate import tabulate
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
import util


def evaluate_and_report(config_model:dict, config_data:dict, split:str = "train") -> None:
    """
    Evaluates a model performance in a specific split of a dataset and show:
        - Report per class
        - Report aggregated on the classes
        - Visual report showing the top most frequent correct and incorrect predictions with Grad Cam included
    :param config_model:
    :param config_data:
    :param split: {train, val, test}
    :return:
    """
    if split not in ["train", "val", "test"]:
        raise ValueError(f"split not recognized: {split}")

    print("\nEvaluating experiment...")

    # Define the folders to be working with
    model_folder = util.config_get_model_dir(model_name=config_model["model_name"])
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    model_dataset_folder = os.path.join(model_folder, config_data["dataset"])
    if not os.path.exists(model_dataset_folder):
        os.mkdir(model_dataset_folder)
    model_dataset_split_folder = os.path.join(model_dataset_folder, split)
    if not os.path.exists(model_dataset_split_folder):
        os.mkdir(model_dataset_split_folder)

    # Get the model
    model = keras.models.load_model(os.path.join(model_folder,"best.hdf5"))

    # Get the dataset
    dataset = load_dataset_split(config_data["dataset"],
                                 split,
                                 config_model["image_size"],
                                 batch_size=1,
                                 shuffle=False,
                                 prefetch=False)

    # In the analysis we are interested in identifying the filenames associated with the images
    # This array will correspond to the other arrays only if the dataset is not shuffled
    image_files = list(dataset.file_paths)

    # ground_truth shape: (m,)
    # predictions has shape (m,)
    # images has shape (m,h,w,c)
    # probabilities has shape (m,K)
    # K being the number of classes
    # We will take a sample of the images of max_number_images, this could be less than the size of the dataset
    # So we will keep an array called "original_index_considered" to map to the filenames
    images, ground_truth, original_index_considered = extract_images_and_groundtruth(dataset,
                                                                                     height=config_model["image_size"][0],
                                                                                     width=config_model["image_size"][1])
    probabilities, predictions = get_probabilities_and_predictions(model, images)

    # Generate and print aggregated performance across classes
    target_names = CLASSES.values()
    clf_report = classification_report(ground_truth, predictions, target_names=target_names)
    with open(os.path.join(model_dataset_split_folder, "classification_report.txt"), "w") as file:
        file.write(str(clf_report))
    print(f"classification report created")

    # Compute most frequent mistakes and true positives
    mistake_list, tp_list = create_mistakes_and_true_positives_count(ground_truth, predictions)
    print_report_with_table(mistake_list, model_dataset_split_folder, "mistakes", topK=10)
    print_report_with_table(tp_list, model_dataset_split_folder, "true positives", topK=10)
    print(f"most frequent mistakes report created")

    # Create the visual report
    create_visual_report(mistake_list,
                         tp_list,
                         images,
                         ground_truth,
                         predictions,
                         probabilities,
                         model,
                         model_dataset_split_folder,
                         image_files,
                         original_index_considered,
                         topK=10)


def print_report_with_table(aggregation_list: list, folder_path: str, report_name: str, topK: int = 10) -> None:
    """
    Table has headers: ["Ranking", "Ground Truth", "Prediction", "Frequency", "% in total predictions"]
    :return: None
    """
    table = []
    final_number = min(topK, len(aggregation_list))
    for i in range(final_number):
        row = aggregation_list[i]
        ranking = row[0]
        gt_class_index = row[1]
        pred_class_index = row[2]
        frequency = row[3]
        gt_class = CLASSES[gt_class_index]
        pred_class = CLASSES[pred_class_index]
        percentage_within_predictions = row[4]
        new_row = [ranking, gt_class, pred_class, frequency, percentage_within_predictions]
        table.append(new_row)
    report = tabulate(table, headers=["Ranking",
                                      "Ground Truth",
                                      "Prediction",
                                      "Frequency",
                                      "% in total predictions"])
    with open(os.path.join(folder_path, report_name + ".txt"), "w") as file:
        file.write(str(report))


def create_mistakes_and_true_positives_count(ground_truth: np.ndarray, predictions: np.ndarray) -> list:
    """
    Creates a list of 5 tuples, such as (2, 4, 5, 10, 0.11) each element representing:
        - Ranking of this frequency event (2)
        - ground truth index class (4)
        - predicted index class (5)
        - frequency (10)
        - % within total predictions (0.11)
    :param ground_truth: (m,) numpy array
    :param predictions: (m,) numpy array
    :return:
    """

    # Both gt and pred must be the same shape
    assert ground_truth.shape == predictions.shape, "ground truth and predictions must have similar shape"
    assert len(ground_truth.shape) == 1, "inputs must be vectors"

    # Retrieve the shapes
    m = ground_truth.shape[0]

    # Joint gt_pred array
    joint_gt_pred = np.column_stack((ground_truth, predictions))

    # Remove correct predictions
    joint_gt_pred_mistakes = joint_gt_pred[joint_gt_pred[:,0] != joint_gt_pred[:,1]]  # gt != pred
    joint_gt_pred_tp = joint_gt_pred[joint_gt_pred[:,0] == joint_gt_pred[:,1]]  # gt == pred

    # Aggregate and count the cases
    unique_mistakes, count_mistakes = np.unique(joint_gt_pred_mistakes, return_counts=True, axis=0)
    unique_tp, count_tp = np.unique(joint_gt_pred_tp, return_counts=True, axis=0)

    # Transform to list and sort in a descending fashion
    gt_pred_mistakes = list(np.column_stack((unique_mistakes, count_mistakes)))
    gt_pred_mistakes.sort(key=lambda x:-x[-1])
    gt_pred_tp = list(np.column_stack((unique_tp, count_tp)))
    gt_pred_tp.sort(key=lambda x: -x[-1])

    # Add the ranking and the percentage
    mistake_list = []
    tp_list = []
    for i, (gt, pred, count) in enumerate(gt_pred_mistakes):
        percentage = round(count / m,2)
        mistake_list.append((i+1, int(gt), int(pred), int(count), percentage))
    for i, (gt, pred, count) in enumerate(gt_pred_tp):
        percentage = round(count / m,2)
        tp_list.append((i+1, int(gt), int(pred), int(count), percentage))

    return mistake_list, tp_list


def get_probabilities_and_predictions(model: keras.Model, images: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """
    Gets the probabilities and the predictions (with argmax) of the model with a given dataset with K classes
    :param model: keras Model
    :param images: of shape (m, h, w, c)
    :param batch_size: batch size to process the images
    :return:
        - probabilities: np array with shape (m, K)
        - predictions: np array with shape (m,)
    """

    # Get the shapes
    m = images.shape[0]
    K = len(CLASSES)

    # Result will be stored here:
    probabilities = np.zeros((m,K))

    # Iterate over the images to process them in batches due to memory constraints
    batches_number = int(math.ceil(m/batch_size))

    for b in range(batches_number):
        print(f"Processing batch {b+1}/{batches_number}")
        index_ini = b*batch_size
        index_end = min(index_ini + batch_size, m)  # edge case for the last batch
        probabilities_batch = model.predict(x=images[index_ini:index_end])
        probabilities[index_ini:index_end,:] = probabilities_batch

    # This is a (m,) vector
    predictions = np.argmax(probabilities, axis=1).astype(int)
    return probabilities, predictions


def extract_images_and_groundtruth(dataset: tf.data.Dataset, height: int, width: int, max_number_images: int = 2000) -> \
        (np.ndarray, np.ndarray, list):
    """
    From a tf.data.Dataset extracts the images and the gt
        - param dataset:
        - param height:
        - param width:
        - param max_number_images:
    :return:
    """

    # Get the size and shapes to fill the output arrays
    m = len(dataset.file_paths)

    # Check if m > max_number_images and create a sample logic
    final_number = m
    consider_image = None
    if m > max_number_images:
        final_number = max_number_images
        consider_image = [True]*final_number + [False]*(m-final_number)
        util.shuffle_list(consider_image)
    else:
        consider_image = [True]*final_number

    # Create the arrays to fill them
    images = np.zeros((final_number, height, width, 3))
    ground_truth = np.zeros((final_number,))

    # Now fill these arrays
    original_index_considered = []
    print("Extracting the images and labels from the dataset")
    i = 0
    for j, (img, gt) in enumerate(dataset):
        if consider_image[j]:
            print(f"example {j}: selected -> {i + 1}/{final_number}")
            original_index_considered.append(j)
            images[i, :, :, :] = img
            ground_truth[i] = int(gt)
            i += 1
        else:
            print(f"example {j}: not selected")

    return images, ground_truth, original_index_considered


def create_visual_report(mistakes: list, true_positives: list, images: np.ndarray, ground_truth: np.ndarray,
                         predictions: np.ndarray, probabilities: np.ndarray, model: keras.Model, save_directory:str,
                         image_files: list, original_index_considered: list, topK: int = 10):

    lists = [mistakes, true_positives]
    for l in lists:
        for i, data in enumerate(l):
            if i == topK:
                break

            # Retrieve the data, for instance:
            # (1, 5, 4, 10, 0.09)
            ranking, gt_index, pred_index, count, percentage = data

            # Connect these cases with the original image index
            indices = np.arange(len(ground_truth))
            joint_index_gt_pred = np.column_stack((indices, ground_truth, predictions)).astype(int)
            joint_index_gt_pred = joint_index_gt_pred[joint_index_gt_pred[:, 1] == gt_index]  # filter by gt
            joint_index_gt_pred = joint_index_gt_pred[joint_index_gt_pred[:, 2] == pred_index]  # filter by pred

            # Create the visual report for this combination of gt_index & pred_index
            create_visual_report_single_page(model,
                                             ranking,
                                             gt_index,
                                             pred_index,
                                             percentage,
                                             images,
                                             probabilities,
                                             joint_index_gt_pred,
                                             save_directory,
                                             image_files,
                                             original_index_considered)


def create_visual_report_single_page(model: keras.Model, ranking: int, gt_index: int, pred_index: int,
                                     fraction: float, images: np.ndarray, probabilities: np.ndarray,
                                     joint_index_gt_pred: np.ndarray, save_directory:str, image_files: list,
                                     original_index_considered: list, size:int = 10):

    # Shuffle the rows to pick a random subset of cases
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
    fig = plt.figure(figsize=(14, 20), constrained_layout=True)
    type_analysis = "true_positives" if gt_index == pred_index else "errors"
    #fig, axs = plt.subplots(final_size, 3, figsize=(14, 20), constrained_layout=True)
    title = f"Examples of ground truth = {gt_class} and prediction = {pred_class}" \
            f"\nRanking: #{ranking} among {type_analysis} and {round(100*fraction,2)}% cases in total predictions"
    fig.suptitle(title, fontsize=10)

    # Assemble the plots of each example
    explainer = GradCAM()
    classes_indices = np.arange(len(CLASSES))
    values = []
    for key, value in CLASSES.items():
        values.append(str(key) + "_" + value)
    x_axis = np.linspace(0, 1.0, 5, endpoint=True)

    # Create the row of subplots for each example
    for i in range(final_size):
        index = joint_index_gt_pred[i, 0]
        image_i = images[index]
        probabilities_i = probabilities[index]
        original_image_index = original_index_considered[index]
        image_filename = image_files[original_image_index]

        # Original image
        ax = fig.add_subplot(final_size, 3, 3*i + 1)
        ax.imshow(image_i.astype("uint8"))
        ax.set_axis_off()
        ax.set_title(f"Filename: {image_filename}", fontsize=9)

        # Grad Cam
        data = ([image_i], None)
        grid = explainer.explain(data, model, class_index=pred_index, image_weight=0.2)
        ax = fig.add_subplot(final_size, 3, 3*i + 2)
        ax.imshow(grid.astype("uint8"))
        ax.set_axis_off()
        ax.set_title("Grad CAM", fontsize=9)

        # Probabilities
        x = CLASSES.values()
        ax = fig.add_subplot(final_size, 3, 3*i + 3)
        barlist = ax.barh(y=classes_indices, width=probabilities_i)
        barlist[gt_index].set_color("green")  # This is the ground truth set to green
        ax.set_yticks(classes_indices, labels=values, fontsize=7)
        ax.set_xticks(ticks=x_axis, fontsize=7)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Probability', fontsize=7)
        ax.set_title("Probability prediction per class", fontsize=9)

    filename = "analysis_" + type_analysis + "_top_" + str(ranking).zfill(2)
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









