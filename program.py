import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("spam.csv", encoding="latin-1", usecols=[0, 1], names=["label", "text"], header=0)

df["text"] = df["text"].apply(lambda x: x.replace('\r\n', ' '))

df["label_num"] = df["label"].map({"ham": 0, "spam": 1})


stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

corpus = []
for i in range(len(df)):
    text = df["text"].iloc[i].lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = df["label_num"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

predictions = clf.predict(X)


df["predicted"] = predictions
df["predicted_label"] = df["predicted"].map({0: "ham", 1: "spam"})


for i in range(len(df)):
    print(f"Email {i+1}:")
    print("Text:     ", df['text'].iloc[i])
    print("Actual:   ", df['label'].iloc[i])
    print("Predicted:", df['predicted_label'].iloc[i])
    print("-" * 50)

print("Overall Model Accuracy:", clf.score(X_test, y_test))