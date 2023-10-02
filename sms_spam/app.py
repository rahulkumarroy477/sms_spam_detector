import streamlit as st
import nltk
import pickle
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
english_stopwords = [
    "a", "about", "above", "after", "again", "against", "ain't", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
    "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "will", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
]


def transform_txt(text):
    # Lower case
    text = text.lower()
    # Tokenization
    text = nltk.word_tokenize(text)
    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
     
    # Removing stop words and punctuations
    text = y[:] # we have to clone the list 
    y.clear()
    for i in text:
        if i not in english_stopwords and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    # stemming --> reducing the words to their stem form
    # ex: jumping --> jump
    
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfid = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam classifier')

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    # 1. Preprocess 
    transformed_sms = transform_txt(input_sms)
    # 2. Vectorize
    vector_input = tfid.transform([transformed_sms])
    # 3. Predict

    result = model.predict(vector_input)[0]
    # 4. Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not spam")