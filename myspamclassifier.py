from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
df=pd.read_csv("emails.csv")
print(df.head())
d={1:"spam",0:"not spam"}
xtrain,xtest,ytrain,ytest=train_test_split(df['text'],df['spam'],test_size=0.1,random_state=0)
Vectorizer=CountVectorizer()
counts=Vectorizer.fit_transform(xtrain)
targets=ytrain
Classifier=MultinomialNB()
Classifier.fit(counts,targets)
example_counts=Vectorizer.transform(xtest)
predictions=Classifier.predict(example_counts)
print("1st 5 Predictions:", predictions[:5])
