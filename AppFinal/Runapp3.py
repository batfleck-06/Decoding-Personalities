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
import openai
from PIL import Image, ImageTk


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
        
        #openAI api code
        self.openai_api_label = ttk.Label(self.frame, text="OpenAI API Key:")
        self.openai_api_label.pack()
        
        self.openai_api_entry = ttk.Entry(self.frame)
        self.openai_api_entry.pack()
        
        self.model_engine_label = ttk.Label(self.frame, text="Select Model Engine:")
        self.model_engine_label.pack()
        
        self.model_engine_var = tk.StringVar()
        self.model_engine_dropdown = ttk.Combobox(self.frame, textvariable=self.model_engine_var,
                                                  values=["text-davinci-003", "text-davinci-002"])
        self.model_engine_dropdown.pack()

        self.model_label = ttk.Label(self.frame, text="Select Model:")
        self.model_label.pack()

        # Dropdown for selecting .pkl files
        self.model_dropdown = ttk.Combobox(self.frame, values=["xgb_model.pkl", "logistic_reg_model.pkl","mlp_model.pkl"])
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




    def display_predictions(self, num_predictions=3):
        input_text = self.entry.get()
        selected_model = self.model_dropdown.get()  # Get selected model file
    
        # Preprocess the input text
        preprocessed_text = self.preprocess_text(input_text)
        
        
        # Use OpenAI API for generating LLM-based personality intent
        openai_api_key = self.openai_api_entry.get()
        openai.api_key = openai_api_key
        
        model_engine = self.model_engine_var.get()
        prompt = "What's the personality type according to MBTI test and also tell about the intent expressed in this tweet : " + input_text
        
        # Generate a response from OpenAI
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        
        openai_response = completion.choices[0].text.strip()
    
        # Get top N predicted personality types and their confidence scores
        top_n_predictions, top_n_probs = self.get_personality_types_ml(preprocessed_text, selected_model, num_predictions)
    
        # Replace with your label mapping
        label_mapping = {
            0: 'ENFJ', 1: 'ENFP', 2: 'ENTJ', 3: 'ENTP', 4: 'ESFJ', 5: 'ESFP', 6: 'ESTJ', 7: 'ESTP',
            8: 'INFJ', 9: 'INFP', 10: 'INTJ', 11: 'INTP', 12: 'ISFJ', 13: 'ISFP', 14: 'ISTJ', 15: 'ISTP'
        }
        

        
        personality_images = {
        'ENFJ': 'enfj.png',
        'ENFP': 'enfp.png',
        'ENTJ': 'entj.png',
        'ENTP': 'entp.png',
        'ESFJ': 'esfj.png',
        'ESFP': 'esfp.png',
        'ESTJ': 'estj.png',
        'ESTP': 'estp.png',
        'INFJ': 'infj.png',
        'INFP': 'infp.png',
        'INTJ': 'intj.png',
        'INTP': 'intp.png',
        'ISFJ': 'isfj.png',
        'ISFP': 'isfp.png',
        'ISTJ': 'istj.png',
        'ISTP': 'istp.png',
            }
    
        # Prepare the result message
        result_message = f"OpenAI Prediction: {openai_response}\n\nTop Predicted Personality Types (ML):\n"
        
        # Create a tkinter window for displaying images
        image_window = tk.Toplevel(self.root)
        
        # Append ML predictions to the result message
        for i in range(num_predictions):
            personality_type = label_mapping[top_n_predictions[i]]
            confidence_score = top_n_probs[i]
            # description = personality_descriptions.get(personality_type, "Description not available")
            # result_message += f"{personality_type}: {confidence_score:.2f} - {description}\n"
            
            # Get the image file path for the personality type
            image_path = personality_images.get(personality_type)
            
            if image_path:
                # Load and display the image using Pillow (PIL)
                image = Image.open(image_path)
                image = image.resize((100, 100))  # Resize the image as needed
                image_tk = ImageTk.PhotoImage(image)
            
                # Create a label to display the image
                image_label = ttk.Label(image_window, image=image_tk)
                image_label.image = image_tk
                image_label.pack()
            
            result_message += f"{personality_type}: {confidence_score:.2f}\n"
    
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
