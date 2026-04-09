import pandas as pd
import numpy as np

import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("fake_or_real_news.csv") 



#cleaning data 
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[a-zA-Z ]','',text)
    return text

df['text']=df['text'].apply(clean_text)


#converted the label fake as 0 and Real News as 1
df['label']=df['label'].map({'REAL':1, 'FAKE':0})


x=df['text']
y=df['label']

x_train, x_test,y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)


vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

model = LogisticRegression()
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))


def predict_news(news):
    news = clean_text(news)
    vec = vectorizer.transform([news])
    result = model.predict(vec)

    if result[0] == 1:
        return "REAL NEWS "
    else:
        return "FAKE NEWS "


print(predict_news("Nobody thinks the alleged affair is the actual reasonÂ McCarthy faced opposition in his quest for the speaker's gavel. Rather, the affair seems to be a tool that his enemies inside the caucus and in the larger movement used against him."))
print(predict_news("Aliens seen in Chennai yesterday night"))