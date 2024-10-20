'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Expanded dataset for better accuracy
data = {
    'symptoms': [
        "Fish not eating and slow", 
        "Fish has white spots", 
        "Fish swimming erratically", 
        "Fish appears healthy", 
        "Fish has red lesions",
        "Fish is active and feeding",   # Healthy
        "Fish has cloudy eyes",         # Diseased
        "Fish has fins rotting",        # Diseased
        "Fish swims normally",          # Healthy
        "Fish has unusual behavior",    # Diseased
        "Fish has bulging eyes",        # Diseased
        "Fish is swimming on the bottom",  # Diseased
        "Fish has clear skin",          # Healthy
        "Fish has no appetite",         # Diseased
        "Fish is darting around",       # Diseased
        "Fish is swimming with others"  # Healthy
    ],
    'disease': [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],  # 0: Healthy, 1: Diseased
    'temperature': [28, 30, 32, 25, 27, 26, 29, 31, 24, 33, 28, 29, 26, 30, 32, 25],  # Sample water temperature data
    'pH': [7.5, 6.8, 6.5, 8.0, 7.0, 7.4, 6.9, 6.7, 7.1, 6.6, 7.5, 6.8, 7.2, 6.7, 7.3, 7.0],   # Sample pH data
    'salinity': [30, 28, 35, 40, 32, 29, 31, 38, 34, 36, 33, 29, 31, 37, 36, 32]        # Sample salinity data
}

# Create DataFrame
df = pd.DataFrame(data)

# Display class distribution to check for imbalance
print("Class distribution in 'disease':", np.bincount(df['disease']))

# Convert the text data into TF-IDF features
tfidf = TfidfVectorizer()
df_tfidf = tfidf.fit_transform(df['symptoms'])

# Combine TF-IDF features with water condition features (temperature, pH, salinity)
X = np.hstack([df_tfidf.toarray(), df[['temperature', 'pH', 'salinity']].values])

# Target variable
y = df['disease']

# Split the dataset into training and testing sets, using stratify to keep class balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=200)  # Increase the number of iterations
model.fit(X_train, y_train)

# Medication suggestions with specific antibiotics and treatments
medication_dict = {
    "Fish not eating and slow": "Treat with **Metronidazole** to address potential parasites; ensure good water quality.",
    "Fish has white spots": "Use **Copper-based treatments** (e.g., Cupramine) to treat Ich (white spot disease).",
    "Fish swimming erratically": "Check water quality; may require **stress relief treatments** such as stress coat.",
    "Fish has red lesions": "Consult a veterinarian; may need **Antibiotics** such as **Oxytetracycline** or antifungal treatments.",
    "Fish has cloudy eyes": "Consider using an antibiotic solution like **Terramycin**; check water parameters.",
    "Fish has fins rotting": "Use **Maracyn** or **Kanamycin** to treat fin rot; improve water quality.",
    "Fish has unusual behavior": "Monitor for disease; consider **API Stress Coat** for anti-stress treatment.",
    "Fish has bulging eyes": "Treat with **Anti-inflammatory medications** like **Maracyn 2** and ensure good water quality.",
    "Fish is swimming on the bottom": "Possible swim bladder disease; consult a vet; may need **Epsom salt baths**.",
    "Fish has no appetite": "Use **appetite stimulants** like **Garlic Guard**; ensure proper diet.",
    "Fish is darting around": "May indicate stress or parasites; check water quality and use **Anti-parasitics**."
}

# Function to predict disease based on user input
def predict_disease(user_input, temperature, pH, salinity):
    # Preprocess user input
    user_input_tfidf = tfidf.transform([user_input]).toarray()
    
    # Combine user input with water conditions
    user_features = np.hstack([user_input_tfidf, [[temperature, pH, salinity]]])
    
    # Predict the disease
    prediction = model.predict(user_features)
    
    # Determine the medication suggestion based on user input
    medication_suggestion = medication_dict.get(user_input, "No specific treatment recommended.")
    
    # Return prediction result and medication suggestion
    return "Diseased" if prediction[0] == 1 else "Healthy", medication_suggestion

# Example usage with user input
user_input = "Fish has cloudy eyes"  # Input indicating disease
temperature = 29                      # Example temperature
pH = 6.9                              # Example pH level
salinity = 33                         # Example salinity level
result, medication = predict_disease(user_input, temperature, pH, salinity)

# Output the prediction and medication suggestion
print(f"Prediction for input '{user_input}': {result}")
print(f"Suggested Medication: {medication}")

# Evaluate the model’s performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the accuracy and confusion matrix
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

'''
