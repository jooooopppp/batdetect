import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from batdetect2 import api
from batdetect2 import plot
import glob

MAX_DURATION = 2
DETECTION_THRESHOLD = 0.3


def make_prediction(file_name, detection_threshold=DETECTION_THRESHOLD):
    # configure the model run
    run_config = api.get_config(
        detection_threshold=detection_threshold,
        max_duration=MAX_DURATION,
    )

    # process the file to generate predictions
    results = api.process_file(file_name, config=run_config)

    # extract the detections
    detections = results["pred_dict"]["annotation"]

    # create a dictionary to store the highest detection probabilities for each species
    species_probs = {}

    # iterate over the detections and store the highest detection probability for each species
    for pred in detections:
        species = pred["class"]
        detection_prob = pred["class_prob"]
        if species in species_probs:
            if detection_prob > species_probs[species]:
                species_probs[species] = detection_prob
        else:
            species_probs[species] = detection_prob

    # create a dataframe of the predictions with the highest detection probabilities
    df = pd.DataFrame(
        [
            {
                "file_name": os.path.basename(file_name),
                "time": pred["start_time"],
                "species": species,
                "detection_prob": detection_prob,
            }
            for species, detection_prob in species_probs.items()
        ]
    )

    im = generate_results_image(file_name, detections, run_config)

    return im, df

def generate_results_image(file_name, detections, config):
    audio = api.load_audio(
        file_name,
        max_duration=config["max_duration"],
        time_exp_fact=config["time_expansion"],
        target_samp_rate=config["target_samp_rate"],
    )

    spec = api.generate_spectrogram(audio, config=config)

    plt.close("all")
    fig = plt.figure(
        1,
        figsize=(15, 4),
        dpi=100,
        frameon=False,
    )
    ax = fig.add_subplot(111)
    plot.spectrogram_with_detections(spec, detections, ax=ax)
    plt.tight_layout()

    # Convert figure to numpy array
    fig.canvas.draw()
    im = np.array(fig.canvas.renderer.buffer_rgba())

    return im

def save_results_image(file_name, image):
    output_folder = "data/output"
    file_name_without_ext = os.path.splitext(os.path.basename(file_name))[0]
    image_path = os.path.join(output_folder, file_name_without_ext + ".png")
    image_path_greyscale = os.path.join(output_folder, file_name_without_ext + "_greyscale.png")

    # Save the color image
    plt.imsave(image_path, image)
    print("Color image saved:", image_path)

    # Convert the image to grayscale
    image_greyscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Save the grayscale image
    plt.imsave(image_path_greyscale, image_greyscale, cmap='gray')
    print("Grayscale image saved:", image_path_greyscale)

def process_audio_files():
    raw_folder = "data/raw"
    output_folder = "data/output"
    audio_files = glob.glob(os.path.join(raw_folder, "*.wav"))

    df_list = []

    for file_path in audio_files:
        im, df = make_prediction(file_path)
        save_results_image(file_path, im)

        # Add the dataframe to the list
        df_list.append(df)

    # Concatenate all the dataframes
    predictions_df = pd.concat(df_list, ignore_index=True)

    # Save the predictions as CSV
    predictions_path = os.path.join(output_folder, "predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print("Predictions saved:", predictions_path)


if __name__ == "__main__":
    process_audio_files()
