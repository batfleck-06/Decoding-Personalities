#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: batfleck06
"""


# Libraries to be imported in order to run the application

import tkinter as tk
from tkinter import ttk
import openai
from PIL import Image, ImageTk, ImageSequence
import squarify
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import numpy as np
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm
import pygame


# Declaring the main appplication class
class PersonalityTypeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Star Wars Personality Predictor")
        self.root.geometry("1440x900")
        
        # Initialising pygame library for playing background audio
        pygame.mixer.init()
        pygame.mixer.music.load("starwars.mp3")  
        pygame.mixer.music.set_volume(0.5)  
        pygame.mixer.music.play(-1) 
        
        # Binding the stop_music method to the application window's close event
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Loading the animated GIF
        self.gif_path = "ahsoka3.gif"
        self.gif = Image.open(self.gif_path)
        self.gif_frames = [self.resize_frame(frame) for frame in ImageSequence.Iterator(self.gif)]
        self.current_frame_index = 0

        # Creating a Label widget to display the animated GIF
        self.bg_label = tk.Label(root)
        self.bg_label.place(relwidth=1, relheight=1)

        # Scheduling the update_animation function to run
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

        # Creating a Text widget with a vertical scrollbar for the input text box
        self.text_widget = tk.Text(self.frame, height=10, width=40, wrap=tk.WORD)
        self.text_scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.text_widget.yview)
        self.text_widget.config(yscrollcommand=self.text_scrollbar.set)
        self.text_widget.grid(row=1, column=0, padx=(0, 10), sticky="w")
        self.text_scrollbar.grid(row=1, column=1, sticky="ns")

        # OpenAI API logic for performing LLM based prediction
        self.openai_api_label = ttk.Label(self.frame, text="OpenAI API Key:")
        self.openai_api_label.grid(row=2, column=0, sticky="w")

        self.openai_api_entry = ttk.Entry(self.frame)
        self.openai_api_entry.grid(row=3, column=0, padx=(0, 10), sticky="w")

        self.model_engine_label = ttk.Label(self.frame, text="Select AI Model:")
        self.model_engine_label.grid(row=4, column=0, sticky="w")

        self.model_engine_var = tk.StringVar()
        self.model_engine_dropdown = ttk.Combobox(self.frame, textvariable=self.model_engine_var,
                                                  values=["text-davinci-003", "text-davinci-002"])
        self.model_engine_dropdown.grid(row=5, column=0, padx=(0, 10), sticky="w")

        self.model_label = ttk.Label(self.frame, text="Select ML Model:")
        self.model_label.grid(row=6, column=0, sticky="w")
        

        # Code for the dropdown for selecting .pkl files
        self.model_dropdown = ttk.Combobox(self.frame, values=["xgb_model.pkl", "logistic_reg_model.pkl", "mlp_model.pkl"])
        self.model_dropdown.grid(row=7, column=0, padx=(0, 10), sticky="w")

        self.predict_button = ttk.Button(self.frame, text="Predict", command=self.display_predictions)
        self.predict_button.grid(row=8, column=0, sticky="w")

        self.lemmatizer = Lemmatizer()
        self.tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        
        bottom_label = ttk.Label(self.root, text="What Star Wars Personality Are you?", font=("Comic Sans MS", 28, "bold"))
        bottom_label.pack(side="bottom", pady=(0, 10))

    def resize_frame(self, frame, width=1440, height=900):
        return ImageTk.PhotoImage(frame.resize((width, height), Image.ANTIALIAS))
    
    
    # Method for stopping the BG music
    def stop_music(self):
        pygame.mixer.music.stop()
        
    def on_closing(self):
        # This function is called when the window is being closed
        self.stop_music()
        self.root.destroy()

    def update_animation(self):
        # Updating the label with the next frame of the animated GIF
        self.bg_label.config(image=self.gif_frames[self.current_frame_index])
        self.current_frame_index = (self.current_frame_index + 1) % len(self.gif_frames)

        # Getting the duration of the current frame from the gif
        frame_duration = self.gif.info.get("duration", 100)  # Default to 100 milliseconds

        # Scheduling the next update_animation call for a smooth animation
        self.root.after(frame_duration, self.update_animation)
        
     # Text cleaning method from the main codebase  
    def clean_text(self, data):
        cleaned_text = []
        for sentence in tqdm(data):
            sentence = sentence.lower()

            # Removing URLs
            sentence = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', sentence)

            # Removing non-alphanumeric characters
            sentence = re.sub('[^0-9a-z]', ' ', sentence)

            cleaned_text.append(sentence)
        return cleaned_text
    
    # Method for preprocessing the input text
    def preprocess_text(self, text):
        cleaned_text = self.clean_text([text])[0]
        lemmatized_text = self.lemmatizer(cleaned_text)
        preprocessed_text = ' '.join(lemmatized_text)
        return preprocessed_text
    
    # Method call for predicting the top n perosnlity types using ML
    def get_personality_types_ml(self, preprocessed_text, model_file, num_predictions=3):
        ml_model = joblib.load(model_file)

        text_features = self.tfidf_vectorizer.transform([preprocessed_text])

        top_n_predictions = ml_model.predict_proba(text_features).argsort()[0][-num_predictions:][::-1]
        top_n_probs = ml_model.predict_proba(text_features)[0][top_n_predictions]

        return top_n_predictions, top_n_probs
    
    
    # Method for the main results window
    def display_predictions(self, num_predictions=3):
        input_text = self.text_widget.get("1.0", "end-1c")  
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

        # Dictionary containinng filenames for each personality type's image
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
        result_window.title("May the Force be with You")

        # Set the window dimensions to match the screen size
        screen_width = 1200
        screen_height = result_window.winfo_screenheight()
        result_window.geometry(f"{screen_width}x{screen_height}")
        
        top_frame = ttk.Frame(result_window)
        top_frame.grid(row=0, column=0, padx=20, pady=20, rowspan=1)
        
        lower_frame = ttk.Frame(result_window)
        lower_frame.grid(row=1, column=0, padx=20, pady=20, rowspan=1)
        
        
        # Open AI results grid layout setup
        openai_frame = ttk.Frame(top_frame, width=300, height=300)
        openai_frame.grid(row=0, column=0, padx=20, pady=20)
        openai_frame.grid_propagate(False)
        
        
        # Machine learning results grid layout setup
        ml_frame = ttk.Frame(top_frame, width=300, height=300)
        ml_frame.grid(row=0, column=1, padx=20, pady=10)
        ml_frame.grid_propagate(False)
        
        ml_graph_frame = ttk.Frame(top_frame, width=400, height=300)
        ml_graph_frame.grid(row=0, column=2, padx=10, pady=10)
        ml_graph_frame.grid_propagate(False)

        openai_label = ttk.Label(openai_frame, text="OpenAI Prediction", font=("Arial", 12))
        openai_label.grid(row=0, column=0, sticky="w")
        openai_text = tk.Text(openai_frame, height=20, width=40, wrap=tk.WORD)
        openai_text.grid(row=1, column=0, padx=(0, 10), sticky="w")
        openai_text.insert(tk.END, openai_response)
        openai_text.config(state=tk.DISABLED)  # Disable text editing

        ml_predictions_label = ttk.Label(ml_frame, text="ML Predicted top 3 personality types", font=("Arial", 12))
        ml_predictions_label.grid(row=0, column=0, sticky="w")
        ml_text = tk.Text(ml_frame, height=4, width=30, wrap=tk.WORD)
        ml_text.grid(row=1, column=0, padx=(0, 10), sticky="w")
        ml_predictions = ""
        for i in range(num_predictions):
            personality_type = label_mapping[top_n_predictions[i]]
            confidence_score = top_n_probs[i]
            ml_predictions += f"ML Prediction {i + 1}: {personality_type} ({confidence_score:.2f})\n"
        ml_text.insert(tk.END, ml_predictions)
        ml_text.config(state=tk.DISABLED)  # Disable text editing

        # Creating a list of labels and sizes for the treemap based on ML results
        treemap_labels = [f"{label_mapping[top_n_predictions[i]]} ({top_n_probs[i]:.2f})" for i in range(num_predictions)]
        treemap_sizes = [int(top_n_probs[i] * 100) for i in range(num_predictions)]

        # Creating a subplot for the treemap
        plt.figure(figsize=(8, 4))
        plt.subplot(121)

        # Creating the treemap using squarify
        squarify.plot(sizes=treemap_sizes, label=treemap_labels, alpha=0.7, color=['red', 'green', 'blue'],
                      text_kwargs={'fontdict': {'weight': 'bold'}})
        plt.axis('off')

        # Setting the title for the treemap
        plt.title("Distribution for top 3 personlities predicted")

        # Creating a FigureCanvasTkAg
        treemap_canvas = FigureCanvasTkAgg(plt.gcf(), master=ml_graph_frame)
        treemap_canvas.get_tk_widget().grid(row=0, column=0, padx=15, pady=0, sticky ="nsew")

        # Extracting keywords from the input text
        extracted_keywords = self.extract_keywords(preprocessed_text)
        
        
        # Creating a label to display the extracted keywords
        keywords_frame = ttk.Frame(ml_frame, width=300, height=150)
        keywords_frame.grid(row=2, column=0, padx=0, pady=0, sticky="w")
        
        keywords_label = ttk.Label(keywords_frame, text="Keywords that influenced the prediction ", font=("Arial", 12))
        keywords_label.grid(row=0, column=0, padx=0, pady=0, sticky="w")
        
        keywords_text = tk.Text(keywords_frame, height=6, width=32, wrap=tk.WORD)
        keywords_text.grid(row=1, column=0, padx=0, pady=10, sticky="w")
        keywords_text.insert(tk.END, "\n".join(extracted_keywords))
        keywords_text.config(state=tk.DISABLED)  # Disable text editing
        
        keywords_frame.grid_propagate(False)


        # Creating a frame for ML prediction images
        ml_images_frame = ttk.Frame(lower_frame,width=900, height=300)
        ml_images_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        ml_images_frame.grid_propagate(False)

        ml_images_label = ttk.Label(ml_images_frame, text="Your Star wars personalities", font=("Arial", 20, "bold"))
        ml_images_label.grid(row=0, column=0, padx=0, pady=0, sticky="w")

        image_frame = tk.Frame(ml_images_frame)
        image_frame.grid(row=1, column=0, padx=0, pady=0, sticky="w")

        # Calculating the number of columns and rows for the image grid
        num_columns = 3  # Adjust the number of columns as needed
        num_rows = (num_predictions + num_columns - 1) // num_columns

        for i in range(num_predictions):
            personality_type = label_mapping[top_n_predictions[i]]
            image_path = personality_images.get(personality_type)

            if image_path:
                image = Image.open(image_path)
                image = image.resize((240, 225), Image.ANTIALIAS)
                image_tk = ImageTk.PhotoImage(image)

                # Calculate row and column for each image in the grid
                row = i // num_columns
                col = i % num_columns

                image_label = tk.Label(image_frame, image=image_tk)
                image_label.image = image_tk
                image_label.grid(row=row, column=col, padx=5, pady=5, sticky="w")


    # Method for user input analysis
    def extract_keywords(self, preprocessed_text, num_keywords=5):
        # Extract top TF-IDF keywords from preprocessed text
        text_features = self.tfidf_vectorizer.transform([preprocessed_text])
        tfidf_scores = text_features.sum(axis=0).A1
        top_keyword_indices = tfidf_scores.argsort()[-num_keywords:][::-1]
        feature_names = np.array(self.tfidf_vectorizer.get_feature_names_out())
        top_keywords = feature_names[top_keyword_indices].tolist()

        return top_keywords

#Lemmatiser class
class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2]

if __name__ == "__main__":
    root = tk.Tk()
    app = PersonalityTypeApp(root)
    root.mainloop()
