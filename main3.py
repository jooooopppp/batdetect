import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

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

# Apply SMOTE for oversampling to address class imbalance
# Apply SMOTE for oversampling to address class imbalance
smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Apply MinMaxScaler for feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=4)  # Choose an appropriate number of components
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Create a parameter grid to search over
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
}

# Initialize GridSearchCV
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')

# Fit the model on the training data
grid_search.fit(X_train_pca, y_train_resampled)

# Get the best parameters and best model from the search
best_params = grid_search.best_params_
best_svm_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_svm_model.predict(X_test_pca)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(
    y_test, y_pred, labels=list(species_mapping.values()), target_names=list(species_mapping.keys())
)

print("Best Parameters:", best_params)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
