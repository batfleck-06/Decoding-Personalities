#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: batfleck06
"""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import joblib
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm


class PersonalityTypeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Personality Type Predictor")
        self.root.geometry("400x400")

        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0", font=("Arial", 12))
        self.style.configure("TButton", background="#007bff", foreground="black", font=("Arial", 12))
        self.style.configure("TCombobox", font=("Arial", 12))

        self.frame = ttk.Frame(root)
        self.frame.pack(padx=20, pady=20)

        self.label = ttk.Label(self.frame, text="Enter your text:")
        self.label.pack()

        self.entry = ttk.Entry(self.frame)
        self.entry.pack()

        self.model_label = ttk.Label(self.frame, text="Select Model:")
        self.model_label.pack()

        # Dropdown for selecting .pkl files
        self.model_dropdown = ttk.Combobox(self.frame, values=["xgb_model.pkl", "logistic_reg_model.pkl"])
        self.model_dropdown.pack()

        self.predict_button = ttk.Button(self.frame, text="Predict", command=self.display_predictions)
        self.predict_button.pack()

        # Initialize the Lemmatizer
        self.lemmatizer = Lemmatizer()
        # Load the TF-IDF vectorizer
        self.tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    def clean_text(self, data):
        cleaned_text = []
        for sentence in tqdm(data):
            sentence = sentence.lower()

            # Remove URLs
            sentence = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', sentence)

            # Remove non-alphanumeric characters
            sentence = re.sub('[^0-9a-z]', ' ', sentence)

            cleaned_text.append(sentence)
        return cleaned_text

    def preprocess_text(self, text):
        cleaned_text = self.clean_text([text])[0]
        lemmatized_text = self.lemmatizer(cleaned_text)
        preprocessed_text = ' '.join(lemmatized_text)
        return preprocessed_text
    
    def get_personality_types_ml(self, preprocessed_text, model_file, num_predictions=3):
        # Load the selected model
        ml_model = joblib.load(model_file)
        
        # Transform the preprocessed text using the TF-IDF vectorizer
        text_features = self.tfidf_vectorizer.transform([preprocessed_text])
    
        # Get top N predictions and their probabilities
        top_n_predictions = ml_model.predict_proba(text_features).argsort()[0][-num_predictions:][::-1]
        top_n_probs = ml_model.predict_proba(text_features)[0][top_n_predictions]
    
        return top_n_predictions, top_n_probs    

    # def get_personality_type_ml(self, preprocessed_text, model_file):
    #     # Load the selected model
    #     ml_model = joblib.load(model_file)
        
    #     # Transform the preprocessed text using the TF-IDF vectorizer
    #     text_features = self.tfidf_vectorizer.transform([preprocessed_text])

    #     # Make predictions
    #     ml_prediction = ml_model.predict(text_features)
    #     return ml_prediction

#     def display_predictions(self):
#         input_text = self.entry.get()
#         selected_model = self.model_dropdown.get()  # Get selected model file

#         # Preprocess the input text
#         preprocessed_text = self.preprocess_text(input_text)

#         # Get ML prediction
#         ml_prediction = self.get_personality_type_ml(preprocessed_text, selected_model)

#         # Replace with your label mapping
#         label_mapping = {
#     0: 'ENFJ',
#     1: 'ENFP',
#     2: 'ENTJ',
#     3: 'ENTP',
#     4: 'ESFJ',
#     5: 'ESFP',
#     6: 'ESTJ',
#     7: 'ESTP',
#     8: 'INFJ',
#     9: 'INFP',
#     10: 'INTJ',
#     11: 'INTP',
#     12: 'ISFJ',
#     13: 'ISFP',
#     14: 'ISTJ',
#     15: 'ISTP'
# }

#         # Get the corresponding personality type label
#         ml_prediction_label = label_mapping[ml_prediction[0]]

#         # Display results in a message box
#         result_message = f"ML Prediction: {ml_prediction_label}"
#         messagebox.showinfo("Predictions", result_message)


    def display_predictions(self, num_predictions=3):
        input_text = self.entry.get()
        selected_model = self.model_dropdown.get()  # Get selected model file
    
        # Preprocess the input text
        preprocessed_text = self.preprocess_text(input_text)
    
        # Get top N predicted personality types and their confidence scores
        top_n_predictions, top_n_probs = self.get_personality_types_ml(preprocessed_text, selected_model, num_predictions)
    
        # Replace with your label mapping
        label_mapping = {
            0: 'ENFJ', 1: 'ENFP', 2: 'ENTJ', 3: 'ENTP', 4: 'ESFJ', 5: 'ESFP', 6: 'ESTJ', 7: 'ESTP',
            8: 'INFJ', 9: 'INFP', 10: 'INTJ', 11: 'INTP', 12: 'ISFJ', 13: 'ISFP', 14: 'ISTJ', 15: 'ISTP'
        }
        
        # Personality type descriptions
        personality_descriptions = {
        'ENFJ': 'The Teacher - Warm, empathetic, and responsible. They strive to help and mentor others.',
        'ENFP': 'The Champion - Energetic, enthusiastic, and creative. They enjoy exploring possibilities.',
        'ENTJ': 'The Commander - Assertive, decisive, and strategic. They are natural leaders.',
        'ENTP': 'The Visionary - Innovative, curious, and adaptable. They love tackling complex challenges.',
        'ESFJ': 'The Caregiver - Friendly, practical, and loyal. They put others\' needs first.',
        'ESFP': 'The Performer - Charismatic, spontaneous, and playful. They love to entertain and be social.',
        'ESTJ': 'The Executive - Organized, responsible, and practical. They excel in leadership roles.',
        'ESTP': 'The Dynamo - Energetic, action-oriented, and resourceful. They thrive on challenges.',
        'INFJ': 'The Counselor - Compassionate, insightful, and creative. They seek deeper connections.',
        'INFP': 'The Idealist - Imaginative, empathetic, and values-driven. They seek authenticity.',
        'INTJ': 'The Architect - Analytical, strategic, and independent. They excel in complex problem-solving.',
        'INTP': 'The Thinker - Logical, curious, and inventive. They love exploring theoretical concepts.',
        'ISFJ': 'The Defender - Reliable, patient, and dedicated. They are committed to supporting others.',
        'ISFP': 'The Artist - Sensitive, adaptable, and creative. They express themselves through art and actions.',
        'ISTJ': 'The Inspector - Practical, detail-oriented, and responsible. They value order and tradition.',
        'ISTP': 'The Virtuoso - Observant, independent, and skilled. They excel in hands-on tasks.',
            }
    
        # Prepare the result message
        result_message = "Top Predicted Personality Types:\n"
        for i in range(num_predictions):
            personality_type = label_mapping[top_n_predictions[i]]
            confidence_score = top_n_probs[i]
            description = personality_descriptions.get(personality_type, "Description not available")
            result_message += f"{personality_type}: {confidence_score:.2f} - {description}\n"
    
        # Display results in a message box
        messagebox.showinfo("Predictions", result_message)

class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2]

if __name__ == "__main__":
    root = tk.Tk()
    app = PersonalityTypeApp(root)
    root.mainloop()
