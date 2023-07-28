import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from batdetect2 import api
from batdetect2 import plot
import glob
import gc

MAX_DURATION = 2
DETECTION_THRESHOLD = 0.3
BATCH_SIZE = 500

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

    fig.canvas.draw()
    im = np.array(fig.canvas.renderer.buffer_rgba())

    return im, spec

def calculate_mean_frequency(spec):
    spec_np = spec.cpu().numpy()  # Convert tensor to Numpy array

    threshold = np.max(spec_np) * 0.1
    mean_frequency = np.mean(np.where(spec_np > threshold)[0])
    return mean_frequency

def calculate_min_frequency(spec):
    spec_np = spec.cpu().numpy()

    threshold = np.max(spec_np) * 0.1
    min_frequency = np.min(np.where(spec_np > threshold)[0])
    return min_frequency

def calculate_max_frequency(spec):
    spec_np = spec.cpu().numpy()

    threshold = np.max(spec_np) * 0.1
    max_frequency = np.max(np.where(spec_np > threshold)[0])
    return max_frequency

def calculate_peak_intensity(spec):
    spec_np = spec.cpu().numpy()

    peak_intensity = np.max(spec_np)
    return peak_intensity


def make_prediction(file_name, detection_threshold=DETECTION_THRESHOLD):
    run_config = api.get_config(
        detection_threshold=detection_threshold,
        max_duration=MAX_DURATION,
    )

    results = api.process_file(file_name, config=run_config)
    detections = results["pred_dict"]["annotation"]

    species_probs = {}
    for pred in detections:
        species = pred["class"]
        detection_prob = pred["class_prob"]
        if species in species_probs:
            if detection_prob > species_probs[species]:
                species_probs[species] = detection_prob
        else:
            species_probs[species] = detection_prob

    im, spec = generate_results_image(file_name, detections, run_config)
    df = pd.DataFrame(
        [
            {
                "file_name": os.path.basename(file_name),
                "time": pred["start_time"],
                "species": species,
                "detection_prob": detection_prob,
                "mean_frequency": calculate_mean_frequency(spec),
                "start_time": pred["start_time"],
                "end_time": pred["end_time"],
                "min_frequency": calculate_min_frequency(spec),
                "max_frequency": calculate_max_frequency(spec),
                "sampling_rate": run_config["target_samp_rate"],
                "peak_intensity": calculate_peak_intensity(spec),
            }
            for species, detection_prob in species_probs.items()
        ]
    )

    return im, df

def save_results_image(file_name, image):
    output_folder = "data/output"
    file_name_without_ext = os.path.splitext(os.path.basename(file_name))[0]
    image_path = os.path.join(output_folder, file_name_without_ext + ".png")
    image_path_greyscale = os.path.join(output_folder, file_name_without_ext + "_greyscale.png")

    os.makedirs(output_folder, exist_ok=True)

    plt.imsave(image_path, image)
    print("Color image saved:", image_path)

    image_greyscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    plt.imsave(image_path_greyscale, image_greyscale, cmap='gray')
    print("Grayscale image saved:", image_path_greyscale)

    plt.close()

def process_audio_files():
    raw_folder = "data/raw"
    output_folder = "data/output"
    audio_files = glob.glob(os.path.join(raw_folder, "*.wav"))

    predictions_df = pd.DataFrame()

    for i in range(0, len(audio_files), BATCH_SIZE):
        batch_files = audio_files[i:i+BATCH_SIZE]

        for file_path in batch_files:
            im, df = make_prediction(file_path)
            save_results_image(file_path, im)

            predictions_df = pd.concat([predictions_df, df], ignore_index=True)

            predictions_path = os.path.join(output_folder, "predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)
            print("Predictions saved:", predictions_path)

        gc.collect()

if __name__ == "__main__":
    process_audio_files()
