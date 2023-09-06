#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 00:11:41 2023

@author: batfleck06
"""

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class KeywordExtractor:
    def __init__(self, tfidf_vectorizer):
        self.tfidf_vectorizer = tfidf_vectorizer

    def extract_keywords(self, preprocessed_text, num_keywords=5):
        """
        Extract top TF-IDF keywords from preprocessed text.

        Args:
            preprocessed_text (str): The preprocessed text.
            num_keywords (int): The number of keywords to extract.

        Returns:
            List[str]: A list of top TF-IDF keywords.
        """
        text_features = self.tfidf_vectorizer.transform([preprocessed_text])
        tfidf_scores = text_features.sum(axis=0).A1
        top_keyword_indices = tfidf_scores.argsort()[-num_keywords:][::-1]
        feature_names = np.array(self.tfidf_vectorizer.get_feature_names_out())
        top_keywords = feature_names[top_keyword_indices].tolist()

        return top_keywords

if __name__ == "__main__":
    # Load the TF-IDF vectorizer that was trained earlier
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    # Create a KeywordExtractor instance
    keyword_extractor = KeywordExtractor(tfidf_vectorizer)

    # Save the KeywordExtractor object to a pickle file
    joblib.dump(keyword_extractor, "keyword_extractor.pkl")