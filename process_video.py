import keras.utils
import CONSTANTS
import cv2
from keras import models
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from tf_explain.core.grad_cam import GradCAM
import util


def extract_frames(video_path: str, width: int, height: int):

    # (1) Create a folder to store the information
    dir_path = os.path.dirname(os.path.realpath(video_path))
    video_folder_path = os.path.join(dir_path, "video_data")
    if not os.path.exists(video_folder_path):
        os.mkdir(video_folder_path)

    # (2) Create a folder to store the frames
    frames_folder_path = os.path.join(video_folder_path, "frames")
    if not os.path.exists(frames_folder_path):
        os.mkdir(frames_folder_path)

    # (3) Extract the frames and save them as .png files
    video = cv2.VideoCapture(video_path)
    frames_number = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    digits_per_file = len(str(frames_number)) + 1  # if its 9999 then we need 5 digits to account for 0 .. 10,000
    if video.isOpened() == False:
        raise ValueError("Error opening video stream or file")
    frame_id = 0
    while video.isOpened():
        print(f"Saved frame {frame_id+1}/{frames_number}")
        return_value, frame = video.read()
        if return_value:
            resized_frame = cv2.resize(frame, (width, height))  # shape: (w, h, c)
            frame_name = str(frame_id).zfill(digits_per_file) + ".png"
            frame_path = os.path.join(frames_folder_path, frame_name)
            cv2.imwrite(frame_path, resized_frame)
            frame_id += 1
        else:
            break
    video.release()


def process_frames(video_folder: str, model: keras.Model) -> None:
    """
    Creates a .csv file with all the predictions and a folder with GRAD cam
    :param video_folder:
    :param model:
    :return:
    """

    # (1) Create a folder to store the grad cam img
    gradcam_folder_path = os.path.join(video_folder, "gradcam")
    if not os.path.exists(gradcam_folder_path):
        os.mkdir(gradcam_folder_path)

    # (2) Get the image files to process
    frames_folder = os.path.join(video_folder, "frames")
    files = [f for f in os.listdir(frames_folder) if f.endswith("png")]  # relative path
    total_files = len(files)

    # (3) Predict and save gradcam image
    explainer = GradCAM()
    data = []  # list like: [file, prob_class_0, prob_class_1, ..., prob_class_n]
    for i,f in enumerate(files):
        print(f"Processing file {i+1}/{total_files}")
        file_path = os.path.join(frames_folder, f)

        # Model prediction
        image_whc = np.asarray(keras.utils.load_img(file_path, grayscale=False, color_mode='rgb'))
        image_1whc = np.expand_dims(image_whc, axis=0)  # the model needs a shape of (1, w, h, c)
        probabilities = model.predict(x=image_1whc)[0]
        data.append([f] + probabilities.tolist())

        # Gradcam processing
        gradcam_data = ([image_whc], None)
        predicted_index_class = np.argmax(probabilities, axis=0).astype(int)
        gradcam_grid = explainer.explain(gradcam_data, model, class_index=predicted_index_class, image_weight=0.2)
        gradcam_img = Image.fromarray(gradcam_grid)
        gradcam_filepath = os.path.join(gradcam_folder_path, f)
        gradcam_img.save(gradcam_filepath)

    # (4) Save the data as .csv
    df = pd.DataFrame(data, columns=["Filepath", "p0","p1","p2","p3","p4","p5","p6","p7","p8","p9"])
    df_filepath = os.path.join(video_folder, "probabilities.csv")
    df.to_csv(df_filepath, index=False)


def process(video, model) -> None:

    WIDTH = 640
    HEIGHT = 480
    resize_dim = (WIDTH, HEIGHT)

    # Check if camera opened successfully
    if video.isOpened() == False:
        raise ValueError("Error opening video stream or file")

    # Establish the figure
    figure = plt.figure()
    plt.ion()
    plt.show()
    x_axis = np.linspace(0, 1.0, 5, endpoint=True)
    ax1 = figure.add_subplot(1, 3, 1)
    ax2 = figure.add_subplot(1, 3, 2)
    ax3 = figure.add_subplot(1, 3, 3)

    # Read until video is completed
    miliseconds_per_frame = 10
    frame_number = 0
    while video.isOpened():
        # Capture frame-by-frame
        return_value, frame = video.read()
        frame_number += 1
        if return_value:

            # Process the frame
            resized_whc = cv2.resize(frame, resize_dim)  # shape: (w, h, c)
            resized_1whc = np.expand_dims(resized_whc, axis=0)  # the model needs a shape of (1, w, h, c)
            probabilities = model.predict(x=resized_1whc)
            predicted_index_class = np.argmax(probabilities, axis=1).astype(int)[0]
            confidence_percentage = int(round(probabilities[0,predicted_index_class]*100,2))
            predicted_label = CONSTANTS.CLASSES[predicted_index_class]
            text = f"{predicted_label} with {confidence_percentage} % confidence"

            # Now plot
            ax1.imshow(resized_whc.astype("uint8"))
            ax1.set_axis_off()
            ax1.set_title(f"Frame # {frame_number}", fontsize=9)
            figure.canvas.draw()
            #figure.canvas.flush_events()


            # Wait a little on each frame
            #cv2.waitKey(miliseconds_per_frame)

        else:
            break

    # When everything done, release the video capture object
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_name = "01_finetune341_RC"
    dataset = "data_noleak_100"
    experiment_dir = util.config_get_experiment_dir(model_name, dataset)
    model = models.load_model(os.path.join(experiment_dir,"best.hdf5"))
    #extract_frames(CONSTANTS.VIDEO_FILE, width=640, height=480)
    video_folder_path = "C:\\Users\\Acer\\Documents\\Datasets\\distracted-driver-detection\\video_data"
    process_frames(video_folder_path, model)
