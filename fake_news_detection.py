import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (only once)
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("fake_news_sample.csv")

# Combine title and text
df['content'] = df['title'] + " " + df['text']

# Preprocess the content


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


df['content'] = df['content'].apply(clean_text)

# Features and labels
X = df['content']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=0)

# Train model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict function


def predict_news(title, text):
    input_text = clean_text(title + " " + text)
    vec = vectorizer.transform([input_text])
    result = model.predict(vec)
    return result[0]


# User input
title_input = input("\nEnter news title: ")
text_input = input("Enter news text: ")

prediction = predict_news(title_input, text_input)
print(f"\n📰 Prediction: {prediction}")
