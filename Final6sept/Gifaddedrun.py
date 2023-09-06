#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 19:47:39 2023

@author: batfleck06
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 16:32:02 2023

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
from PIL import Image, ImageTk,ImageSequence

class PersonalityTypeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Personality Type Predictor")
        self.root.geometry("1440x900")
        
        
        # Load the animated GIF
        self.gif_path = "/Users/batfleck06/Documents/Final Project/Notebooks/New test/AppFinal/ahsoka3.gif"
        self.gif = Image.open(self.gif_path)
        self.gif_frames = [self.resize_frame(frame) for frame in ImageSequence.Iterator(self.gif)]
        self.current_frame_index = 0

        # Create a Label widget to display the animated GIF
        self.bg_label = tk.Label(root)
        self.bg_label.place(relwidth=1, relheight=1)

        # Schedule the update_animation function to run
        self.update_animation()

        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0", font=("Arial", 12))
        self.style.configure("TButton", background="#007bff", foreground="black", font=("Arial", 12))
        self.style.configure("TCombobox", font=("Arial", 12))

        self.frame = ttk.Frame(root)
        self.frame.pack(padx=20, pady=20)

        self.label = ttk.Label(self.frame, text="Enter your text:")
        self.label.grid(row=0, column=0, sticky="w")

        # Create a Text widget with vertical scrollbar
        self.text_widget = tk.Text(self.frame, height=10, width=40, wrap=tk.WORD)
        self.text_scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.text_widget.yview)
        self.text_widget.config(yscrollcommand=self.text_scrollbar.set)
        self.text_widget.grid(row=1, column=0, padx=(0, 10), sticky="w")
        self.text_scrollbar.grid(row=1, column=1, sticky="ns")
        
        # OpenAI API code
        self.openai_api_label = ttk.Label(self.frame, text="OpenAI API Key:")
        self.openai_api_label.grid(row=2, column=0, sticky="w")
        
        self.openai_api_entry = ttk.Entry(self.frame)
        self.openai_api_entry.grid(row=3, column=0, padx=(0, 10), sticky="w")
        
        self.model_engine_label = ttk.Label(self.frame, text="Select Model Engine:")
        self.model_engine_label.grid(row=4, column=0, sticky="w")
        
        self.model_engine_var = tk.StringVar()
        self.model_engine_dropdown = ttk.Combobox(self.frame, textvariable=self.model_engine_var,
                                                  values=["text-davinci-003", "text-davinci-002"])
        self.model_engine_dropdown.grid(row=5, column=0, padx=(0, 10), sticky="w")

        self.model_label = ttk.Label(self.frame, text="Select Model:")
        self.model_label.grid(row=6, column=0, sticky="w")

        # Dropdown for selecting .pkl files
        self.model_dropdown = ttk.Combobox(self.frame, values=["xgb_model.pkl", "logistic_reg_model.pkl", "mlp_model.pkl"])
        self.model_dropdown.grid(row=7, column=0, padx=(0, 10), sticky="w")

        self.predict_button = ttk.Button(self.frame, text="Predict", command=self.display_predictions)
        self.predict_button.grid(row=8, column=0, sticky="w")

        self.lemmatizer = Lemmatizer()
        self.tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    def resize_frame(self, frame, width=1440, height=900):
        return ImageTk.PhotoImage(frame.resize((width, height), Image.ANTIALIAS))        
        
    def update_animation(self):
        # Update the Label with the next frame of the animated GIF
        self.bg_label.config(image=self.gif_frames[self.current_frame_index])
        self.current_frame_index = (self.current_frame_index + 1) % len(self.gif_frames)

        # Get the duration of the current frame
        frame_duration = self.gif.info.get("duration", 100)  # Default to 100 milliseconds

        # Schedule the next update_animation call
        self.root.after(frame_duration, self.update_animation)

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
        ml_model = joblib.load(model_file)
        
        text_features = self.tfidf_vectorizer.transform([preprocessed_text])
    
        top_n_predictions = ml_model.predict_proba(text_features).argsort()[0][-num_predictions:][::-1]
        top_n_probs = ml_model.predict_proba(text_features)[0][top_n_predictions]
    
        return top_n_predictions, top_n_probs

    def display_predictions(self, num_predictions=3):
        input_text = self.text_widget.get("1.0", "end-1c")  # Get text from Text widget
        selected_model = self.model_dropdown.get()
    
        preprocessed_text = self.preprocess_text(input_text)
        
        openai_api_key = self.openai_api_entry.get()
        openai.api_key = openai_api_key
        
        model_engine = self.model_engine_var.get()
        prompt = "What's the personality type according to MBTI test and also tell about the intent expressed in this tweet : " + input_text
        
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        
        openai_response = completion.choices[0].text.strip()
    
        top_n_predictions, top_n_probs = self.get_personality_types_ml(preprocessed_text, selected_model, num_predictions)
    
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
    
        result_window = tk.Toplevel(self.root)
        result_window.title("Predictions")

        # Set the window dimensions to match the screen size
        screen_width = result_window.winfo_screenwidth()
        screen_height = result_window.winfo_screenheight()
        result_window.geometry(f"{screen_width}x{screen_height}")

        openai_frame = ttk.Frame(result_window)
        openai_frame.grid(row=0, column=0, padx=20, pady=40, sticky="nsew")
        
        ml_frame = ttk.Frame(result_window)
        ml_frame.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")
        
        openai_label = ttk.Label(openai_frame, text="OpenAI Prediction", font=("Arial", 12))
        openai_label.grid(row=0, column=0, sticky="w")
        openai_text = tk.Text(openai_frame, height=20, width=40, wrap=tk.WORD)
        openai_text.grid(row=1, column=0, padx=(0, 10), sticky="w")
        openai_text.insert(tk.END, openai_response)
        openai_text.config(state=tk.DISABLED)  # Disable text editing
        
        ml_predictions_label = ttk.Label(ml_frame, text="ML Predictions", font=("Arial", 12))
        ml_predictions_label.grid(row=0, column=0, sticky="w")
        ml_text = tk.Text(ml_frame, height=5, width=30, wrap=tk.WORD)
        ml_text.grid(row=1, column=0, padx=(0, 10), sticky="w")
        ml_predictions = ""
        for i in range(num_predictions):
            personality_type = label_mapping[top_n_predictions[i]]
            confidence_score = top_n_probs[i]
            ml_predictions += f"ML Prediction {i + 1}: {personality_type} ({confidence_score:.2f})\n"
        ml_text.insert(tk.END, ml_predictions)
        ml_text.config(state=tk.DISABLED)  # Disable text editing
    
        # Create image labels for ML predictions and pack them in the ML frame
        for i in range(num_predictions):
            personality_type = label_mapping[top_n_predictions[i]]
            image_path = personality_images.get(personality_type)
    
            if image_path:
                image = Image.open(image_path)
                image = image.resize((240, 225))
                image_tk = ImageTk.PhotoImage(image)
    
                image_label = ttk.Label(ml_frame, image=image_tk)
                image_label.image = image_tk
                image_label.grid(row=i + 2, column=0, padx=10, pady=10)

class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2]

if __name__ == "__main__":
    root = tk.Tk()
    app = PersonalityTypeApp(root)
    root.mainloop()




