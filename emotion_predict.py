import streamlit as st
import pandas as pd
import pickle
import numpy as np
import nltk



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
import string
from PIL import Image

model = pickle.load(open('emotion_model', 'rb'))

nltk.download('stopwords')

st.title("Emotion Predict App")
st.header("Tweet Emotion Prediction")
st.write("This web app predicts the people's emotions based on their tweets")

text_pred = st.text_input("Please enter tweet in the text below")
df = pd.read_csv("mldata.csv" )


def cleaning_text(text):
    stop_words = stopwords.words("english")

    # removing urls from tweets
    text = re.sub(r'http\S+', " ", text)    
    # remove mentions
    text = re.sub(r'@\w+',' ',text)         
    # removing hastags
    text = re.sub(r'#\w+', ' ', text)       
    # removing html tags
    text = re.sub('r<.*?>',' ', text)       
    
    # removing stopwords stopwords 
    text = text.split()
    text = " ".join([word for word in text if not word in stop_words])

    for punctuation in string.punctuation:
        text = text.replace(punctuation, "")
    
    return text

df['Text'] = df['Text'].apply(lambda x: cleaning_text(x)) 
new_text = cleaning_text(text_pred)

encoder = LabelEncoder()
df['Label'] = encoder.fit_transform(df['Emotion'])
  
def tokenizer(x_train, y_train, newv, max_len_word):
    # because the data distribution is imbalanced, "stratify" is used
    X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                      test_size=.2, shuffle=True, 
                                                      stratify=y_train, random_state=0)

    # Tokenizer
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    sequence_dict = tokenizer.word_index
    word_dict = dict((num, val) for (val, num) in sequence_dict.items())

    # Sequence data
    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_padded = pad_sequences(train_sequences,
                                 maxlen=max_len_word,
                                 truncating='post',
                                 padding='post')
    X_val[len(X_val)] = newv
    val_sequences = tokenizer.texts_to_sequences(X_val)
    val_padded = pad_sequences(val_sequences,
                                maxlen=max_len_word,
                                truncating='post',
                                padding='post', )
   

    print(train_padded.shape)
    print(val_padded.shape)
    print('Total words: {}'.format(len(word_dict)))
    return train_padded, val_padded, y_train, y_val, word_dict

X_train, X_val, y_train, y_val, word_dict = tokenizer(df.Text, df.Label, new_text, 100)

label_back = encoder.classes_

if st.button('Predict Overall Performance'):
	pred = model.predict(X_val)[4292]
	pred = np.argmax(pred, axis = 0)
	st.write("The predicted emotion is",label_black[pred])
  	
# 	st.write("The overall predicted score for the above player is", clubs.index(club))
else:
	st.write('Thank You For Trusting Us')
  
