# Tweet Sentiment Analyzer

# Importing reqd. Libraries
import tweepy
import re
import pickle
from tweepy import OAuthHandler

# Initializing the keys
consumer_Key = "yvbuMJBbcmpxmaRFFEiO8ahGG"
consumer_Secret = "tSkASqSxTHyldE2wNTP0KzKuqiIeRN6IwiXXyCx8v3YDYTgZ8J" 
access_Token = "88430729-9gJvDdC1NFF9xEmvTHxnyUSlHEqMjqbvHwQHtgQVH"
access_Secret =  "eDrYHLw44yfuvpSpz01GJadeVcC8QuIqQY1creLvk5hM3"

# Client Authentciation
auth = OAuthHandler(consumer_Key, consumer_Secret)
auth.set_access_token(access_Token, access_Secret)
args = ["climate change"] # Search Token
api = tweepy.API(auth, timeout = 10)

# Fetching real-time Tweets
list_Tweets = []
query = args[0]

if len(args) == 1: # To ensure a single search parameter
    for status in tweepy.Cursor(api.search, q=query+" -filter:retweets", lang="en", result_type="recent").items(100):
        list_Tweets.append(status.text)
        
# Loading Tf-Idf Model and Logistic Regression Classifier 
with open("tfidf_vectorizer.pickle", "rb") as f: # Unpickling the TfidfVectorizer 
    tfidf = pickle.load(f)
with open("classifier.pickle", "rb") as f: # Unpickling the Classifier
    clf = pickle.load(f)


#Pre-processing the Tweets
total_Positive = 0    
total_Negative = 0    
    
for tweet in list_Tweets:
    tweet = re.sub(r"^http://t.co/[a-zA-Z0-9]*\s", " ", tweet) # Removing links from the start of a tweet
    tweet = re.sub(r"\s+http://t.co/[a-zA-Z0-9]*\s", " ", tweet) # Removing links from middle of a tweet
    tweet = re.sub(r"\s+http://t.co/[a-zA-Z0-9]*$", " ", tweet) # Removing links from end of a tweet
    tweet = tweet.lower() # Converting tweets into lowercase
    # Expanding the shorter terms
    tweet = re.sub(r"there's", "there is", tweet)
    tweet = re.sub(r"that's", "that is", tweet)
    tweet = re.sub(r"what's", "what is", tweet)
    tweet = re.sub(r"where's", "where is", tweet)
    tweet = re.sub(r"it's", "it is", tweet)
    tweet = re.sub(r"who's", "who is", tweet)
    tweet = re.sub(r"i'm", "i am", tweet)
    tweet = re.sub(r"they're", "they are", tweet)
    tweet = re.sub(r"who're", "who are", tweet)
    tweet = re.sub(r"ain't", "am not", tweet)
    tweet = re.sub(r"would'nt", "would not", tweet)
    tweet = re.sub(r"should'nt", "should not", tweet)
    tweet = re.sub(r"can't", "can not", tweet)
    tweet = re.sub(r"could't", "could not", tweet)
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"\W", " ", tweet) # Removing non-word characters
    tweet = re.sub(r"\d", " ", tweet) # Removing digits
    tweet = re.sub(r"^[a-z]\s+", " ", tweet) # Removing single character from the start of a tweet
    tweet = re.sub(r"\s+[a-z]\s+", " ", tweet) # Removing single character from the middle of a tweet
    tweet = re.sub(r"^\s+[a-z]$", " ", tweet) # Removing single character from the end of a tweet
    tweet = re.sub(r"\s+", " ", tweet) # Replacing a multi-spaces by a single space
    # Predicting the Sentiment of the Tweets
    sentiment = clf.predict(tfidf.transform([tweet]).toarray())
    print(tweet, ":", sentiment)
    if sentiment[0] == 1:
        total_Positive += 1
    else:
        total_Negative +=1
             
# Plotting the Results
import numpy as np
import matplotlib.pyplot as plt
objects = ["Positive", "Negative"]
y_Pos = np.arange(len(objects))
plt.bar(y_Pos, [total_Positive, total_Negative], alpha=0.5, color="blue", edgecolor="red")
plt.xticks(y_Pos, objects)
plt.ylabel("Number")
plt.title("Number of Positive and Negative Tweets")
plt.show()