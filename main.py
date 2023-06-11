from utils.preprocessing import read_wav_file, bandpass_filter, transform_data
from utils.model_loading import load_model
from utils.analysis import postprocess_output, visualize_results
import torch
import os

# Load the pre-trained model
model_path = './models/Net2DFast_UK_same.pth.tar'  # Replace with your actual model file

# Check if model file exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"No model file found at {model_path}")

model = load_model(model_path)  # Ensure this function correctly loads your model

# Get a list of all the raw audio files in your 'data/raw' directory
audio_files = [f for f in os.listdir('./data/raw') if f.endswith('.wav')]

# Loop over each audio file
for audio_file in audio_files:
    # Read the audio file
    try:
        samplerate, data = read_wav_file(f'./data/raw/{audio_file}')
    except Exception as e:
        print(f"Error reading file {audio_file}: {e}")
        continue

    # Use the bandpass filter on your data
    filtered_data = bandpass_filter(data, lowcut=20000, highcut=100000, fs=samplerate)

    # Transform your filtered data to the correct format for the model
    transformed_data = transform_data(filtered_data)  # Ensure this function correctly formats your data

    # Now use the loaded model to make predictions
    model.eval()  # Put the model in evaluation mode
    with torch.no_grad():  # Don't calculate gradients for speed
        predictions = model(transformed_data)

    # Post-process the model's output
    postprocessed_output = postprocess_output(predictions)  # Ensure this function correctly processes your predictions

    # Visualize the results
    visualize_results(postprocessed_output)  # Ensure this function correctly visualizes your results

# ... rest of your analysis ...
