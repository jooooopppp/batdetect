import pandas as pd
import os

# Function to arrange species in CSV file
def arrange_species_in_csv(input_csv_path, output_csv_path):
    # Load the original CSV file
    original_data = pd.read_csv(input_csv_path)

    # Group by file_name and species, and calculate the maximum detection_prob for each species in each file
    grouped_data = original_data.groupby(['file_name', 'species'])['detection_prob'].max().reset_index()

    # Pivot the data to have species as columns and file_name as index
    arranged_data = grouped_data.pivot(index='file_name', columns='species', values='detection_prob').reset_index()

    # Fill missing values with 0 (no detection)
    arranged_data = arranged_data.fillna(0)

    # Save the arranged data to a new CSV file
    arranged_data.to_csv(output_csv_path, index=False)
    print("Arranged data saved to:", output_csv_path)

# Specify paths
input_csv_path = '/Users/josna/Documents/GitHub/batdetect/data/output/combined_predictions.csv'  # Change this to the actual path of your input CSV
output_csv_path = '/Users/josna/Documents/GitHub/batdetect/data/output/arranged_predictions.csv'  # Change this to the desired output path

# Call the function
arrange_species_in_csv(input_csv_path, output_csv_path)
