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
from tensorflow.keras import layers, callbacks
from tensorflow.keras import Model, Sequential

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
import string
from PIL import Image
from smart_open import smart_open
# model = pickle.load(open('final_model.save.pkl', 'rb'))

nltk.download('stopwords')

st.title("Emotion Predict App")
st.header("Tweet Emotion Prediction")
st.write("This web app predicts the people's emotions based on their tweets")

text_pred = st.text_input("Please enter tweet in the text below")


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

  
new_text = cleaninh_text(text_pred)
  
ntokenizer = Tokenizer(num_words=5)
ntokenizer.fit_on_texts(new_text)
sequence_dict = ntokenizer.word_index
word_dict = dict((num, val) for (val, num) in sequence_dict.items())

    # Sequence data
train_sequences = ntokenizer.texts_to_sequences(new_text)
train_padded = pad_sequences(train_sequences,
                                 maxlen=8,
                                 truncating='post',
                                 padding='post')

if st.button('Predict Overall Performance'):
	st.write(train_padded)
  	
# 	st.write("The overall predicted score for the above player is", clubs.index(club))
else:
	st.write('Thank You For Trusting Us')
  
