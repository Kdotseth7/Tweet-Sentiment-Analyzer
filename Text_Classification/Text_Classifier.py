# Text Classifier

# Importing reqd. Libraries
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files # Used to load text files
nltk.download("stopwords")

# Importing dataset
reviews = load_files("txt_sentoken/")
X, y = reviews.data, reviews.target # neg - 0 & pos - 1

# Pickling the Dataset [Persisting the dataset] 
with open("X.pickle", "wb") as f:
    pickle.dump(X, f)                                                  
with open("y.pickle", "wb") as f:
    pickle.dump(y, f) 
    
del X
del y

# Unpickling the Dataset
with open("X.pickle", "rb") as f:
    X = pickle.load(f)
with open("y.pickle", "rb") as f:
    y = pickle.load(f)
    
# Pre-processing the dataset and Creating the corpus 
corpus = []
for i in range(len(X)):
    review = re.sub(r"\W", " ", str(X[i])) # Removing non-word characters
    review = review.lower() # Converting into lower case
    review = re.sub(r"\s+[a-z]\s+", " ", review) # Removing single characters in the corpus 
    review = re.sub(r"^[a-z]\s+", " ", review) # Removing single characters that occur in the start of a sentence
    review = re.sub(r"\s+", " ", review) # Removing multi-spaces by a single space
    corpus.append(review)
    
# APPROACH-1 : (a.) Bag of Words(BoW) Model
from sklearn.feature_extraction.text import CountVectorizer
cVectorizer = CountVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words("english"))
X = cVectorizer.fit_transform(corpus).toarray()

# APPROACH-1 : (b.) Converting BoW Model into TF-IDF Model
from sklearn.feature_extraction.text import TfidfTransformer
tTransfomer = TfidfTransformer()
X = tTransfomer.fit_transform(X).toarray()

# APPROACH-2 : Creating TF-IDF Model using TfidfVectrizer directly
from sklearn.feature_extraction.text import TfidfVectorizer
tVectorizer = TfidfVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words("english"))
X = tVectorizer.fit_transform(corpus).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
text_Train, text_Test, sent_Train, sent_Test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting classifier to the Training set
from sklearn.linear_model import LogisticRegression
logisiticClassifier = LogisticRegression()
logisiticClassifier.fit(text_Train, sent_Train)

# Predicting the Test set results
sent_Pred = logisiticClassifier.predict(text_Test)

# Making the Confusion Matrix(Classification Evaluation Metric)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_Test, sent_Pred)
Accuracy_Rate = (168+171)/400*100
Error_Rate = 100-Accuracy_Rate

# Pickling or saving our Classifier
with open("classifier.pickle", "wb") as f:
    pickle.dump(logisiticClassifier, f)

# Pickling or saving our TfidfVectorizer
with open("tfidf_vectorizer.pickle", "wb") as f:
    pickle.dump(tVectorizer, f)
    


