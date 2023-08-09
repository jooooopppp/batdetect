import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier # FNN
from sklearn.metrics import classification_report, accuracy_score

# Load the predictions DataFrame
predictions_df = pd.read_csv("data/output/combined_predictions.csv")

# Ensure correct species mapping and encoding
species_mapping = {species: idx for idx, species in enumerate(predictions_df["species"].unique())}
predictions_df["species_encoded"] = predictions_df["species"].map(species_mapping)

# Feature selection (choose relevant columns for classification)
feature_columns = [
    "mean_frequency", "min_frequency", "max_frequency",
    "sampling_rate", "peak_intensity", "detection_prob"
]
X = predictions_df[feature_columns]

# Label and target setup
y = predictions_df["species_encoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for FNNs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the FNN model
fnn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
fnn_model.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = fnn_model.predict(X_test_scaled)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
# Specify the labels parameter in classification_report
classification_rep = classification_report(
    y_test, y_pred, labels=list(species_mapping.values()), target_names=list(species_mapping.keys())
)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
