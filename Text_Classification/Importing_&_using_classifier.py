# Importing and using our model

# Importing reqd. Libraries
import pickle

# Unpickling the Classifier
with open("classifier.pickle", "rb") as f:
    clf = pickle.load(f)
    
# Unpickling the TfidfVectorizer    
with open("tfidf_vectorizer.pickle", "rb") as f:
    tfidf = pickle.load(f)
    
sample = ["You are a nice person, have a good life man"]
sample = tfidf.transform(sample).toarray()
print("Classes : ", end=" ")
print("Positive->1 | Negative->0")
print("Class of Movie Review : %s" % clf.predict(sample))