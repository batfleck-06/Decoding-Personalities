#!/usr/bin/env python
# coding: utf-8

# In[1]:


# for data analysis
import pandas as pd
import numpy as np


#For data Modeling
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from category_encoders import TargetEncoder
import torch
import torch.nn as nn
import torch.optim as optim



#For NLP
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


#Visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud


#Miscellaneous

# for progress  bars
from tqdm import tqdm

#regular expressions
import re

#for .pkl file
import joblib


# In[2]:


nltk.download('wordnet')


# In[3]:


df = pd.read_csv("mbti.csv") 
df.head(5)


# In[4]:


# Calculate the target sample size per class
total_samples = df.shape[0]
num_classes = len(df['type'].unique())
target_samples_per_class = total_samples // num_classes


# In[5]:


target_samples_per_class


# In[6]:


total_samples


# In[7]:


num_classes


# In[8]:


# Dictionary to store posts for each personality type
personality_posts = {ptype: [] for ptype in df['type'].unique()}

# Iterate through each row and populate the dictionary
for index, row in df.iterrows():
    personality_posts[row['type']].append(row['posts'])

# Lists to store balanced data
balanced_features = []
balanced_labels = []


# In[9]:


# Iterate through each personality type
for personality_type, posts in personality_posts.items():
    num_samples = len(posts)
    
    if num_samples >= target_samples_per_class:
        # Sample random indices
        sampled_indices = np.random.choice(num_samples, target_samples_per_class, replace=False)
        
        # Add the sampled data to the balanced sets
        balanced_features.extend([posts[i] for i in sampled_indices])
        balanced_labels.extend([personality_type] * target_samples_per_class)
    else:
        # If fewer posts than target_samples_per_class, use all available posts
        balanced_features.extend(posts)
        balanced_labels.extend([personality_type] * num_samples)


# In[10]:


# Shuffle the data
shuffled_indices = np.random.permutation(len(balanced_features))
balanced_features = [balanced_features[i] for i in shuffled_indices]
balanced_labels = [balanced_labels[i] for i in shuffled_indices]

# Split into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    balanced_features, balanced_labels, test_size=0.2, random_state=42
)


# In[11]:


def clean_text(data):
    data_length = []
    lemmatizer = WordNetLemmatizer()
    cleaned_text = []
    for sentence in tqdm(data):
        sentence = sentence.lower()

        # Remove URLs
        sentence = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', sentence)

        # Remove non-alphanumeric characters
        sentence = re.sub('[^0-9a-z]', ' ', sentence)

        data_length.append(len(sentence.split()))
        cleaned_text.append(sentence)
    return cleaned_text, data_length


# In[12]:


# Clean train and test features
cleaned_train_features, train_data_lengths = clean_text(train_features)


# In[13]:


cleaned_test_features, test_data_lengths = clean_text(test_features)


# In[14]:


class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2]

# Initialize the Lemmatizer
lemmatizer = Lemmatizer()


# In[15]:


# Lemmatize cleaned train and test features
lemmatized_train_features = [lemmatizer(sentence) for sentence in cleaned_train_features]
lemmatized_test_features = [lemmatizer(sentence) for sentence in cleaned_test_features]

# Convert lemmatized features back to sentences
lemmatized_train_sentences = [' '.join(sentence) for sentence in lemmatized_train_features]
lemmatized_test_sentences = [' '.join(sentence) for sentence in lemmatized_test_features]

# Initialize the vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')


# In[16]:


# Fit the vectorizer on lemmatized training data and transform training and testing data
train_post = vectorizer.fit_transform(lemmatized_train_sentences)
test_post = vectorizer.transform(lemmatized_test_sentences)
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


# In[17]:


label_encoder = LabelEncoder()

# Fit and transform labels for both training and testing sets
train_target = label_encoder.fit_transform(train_labels)
test_target = label_encoder.transform(test_labels)
# Save the label encoder for later use
joblib.dump(label_encoder, "label_encoder.pkl")


# In[18]:


# Get the mapping between encoded labels and original labels
encoded_to_original_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
encoded_to_original_mapping


# In[19]:


models_accuracy={}


# In[21]:


model_xgb=XGBClassifier(max_depth=5, n_estimators=50, learning_rate=0.1)
model_xgb.fit(train_post,train_target)


# In[22]:


print('train classification report \n ',classification_report(train_target,model_xgb.predict(train_post),target_names=label_encoder.inverse_transform([i for i in range(16)])))
print('test classification report \n ',classification_report(test_target,model_xgb.predict(test_post),target_names=label_encoder.inverse_transform([i for i in range(16)])))


# In[23]:


models_accuracy['XGBoost Classifier']=accuracy_score(test_target,model_xgb.predict(test_post))


# In[24]:


joblib.dump(model_xgb, "ml_model.pkl")


# In[20]:


#Logistic Reg


# In[21]:


model_log=LogisticRegression(max_iter=3000,C=0.5,n_jobs=-1)
model_log.fit(train_post,train_target)
joblib.dump(model_log, "ml_model.pkl")


# In[22]:


print('train classification report \n ',classification_report(train_target,model_log.predict(train_post),target_names=label_encoder.inverse_transform([i for i in range(16)])))


# In[23]:


models_accuracy['logistic regression']=accuracy_score(test_target,model_log.predict(test_post))


# In[24]:


accuracy=pd.DataFrame(models_accuracy.items(),columns=['Models','Test accuracy'])
accuracy


# In[25]:


# Load the saved model
ml_model = joblib.load("ml_model.pkl")

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocess the random text
random_text = "That's another silly misconception. That approaching is logically is going to be the key to unlocking whatever it is you think you are entitled to.   Nobody wants to be approached with BS"


# Clean the random text
cleaned_random_text, _ = clean_text([random_text])

# Lemmatize the cleaned random text
lemmatized_random_text = lemmatizer(cleaned_random_text[0])

# Convert lemmatized random text back to a sentence
lemmatized_random_text_sentence = ' '.join(lemmatized_random_text)

# Transform the lemmatized random text using the TF-IDF vectorizer
text_features = tfidf_vectorizer.transform([lemmatized_random_text_sentence])

# Make predictions
ml_prediction = ml_model.predict(text_features)
ml_prediction_label = label_encoder.inverse_transform(ml_prediction)[0]

print("Predicted personality type:", ml_prediction_label)


# In[ ]:


# Convert data to PyTorch tensors
train_data_tensor = torch.Tensor(train_post.toarray())
train_target_tensor = torch.LongTensor(train_target)
test_data_tensor = torch.Tensor(test_post.toarray())
test_target_tensor = torch.LongTensor(test_target)


# In[33]:


# Define a simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Assuming batch_first=True
        return out

# Hyperparameters
input_size = train_data_tensor.shape[1]
hidden_size = 64
output_size = len(label_encoder.classes_)
learning_rate = 0.001
num_epochs = 10
batch_size = 64

# Initialize the RNN model
rnn_model = SimpleRNN(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(train_data_tensor), batch_size):
        inputs = train_data_tensor[i:i+batch_size]
        targets = train_target_tensor[i:i+batch_size]
        
        # Forward pass
        outputs = rnn_model(inputs)
        loss = criterion(outputs, targets)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained RNN model
torch.save(rnn_model.state_dict(), "trained_rnn_model.pth")

# Load the trained RNN model
loaded_rnn_model = SimpleRNN(input_size, hidden_size, output_size)
loaded_rnn_model.load_state_dict(torch.load("trained_rnn_model.pth"))

# Make predictions using the trained RNN model
with torch.no_grad():
    test_outputs = loaded_rnn_model(test_data_tensor)
    _, predicted = torch.max(test_outputs, 1)
    predicted_labels = label_encoder.inverse_transform(predicted.numpy())

print("Predicted personality types:", predicted_labels)


# In[ ]:




