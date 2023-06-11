# utils/classification.py
import tensorflow as tf
from sklearn.model_selection import train_test_split

def create_model(input_shape, num_species):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(num_species, activation='softmax')
    ])
    return model

def train_model(model, features, labels, epochs=10):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(features_train, labels_train, epochs=epochs)

    loss, accuracy = model.evaluate(features_test, labels_test)
    print(f'Accuracy: {accuracy}')

    return model
