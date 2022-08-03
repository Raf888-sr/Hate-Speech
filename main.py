import streamlit as st
import pickle
import pandas as pd
import numpy as np
import json
import re
import plotly.express as px
import unicodedata
import contractions
import nltk
import tensorflow  as tf
from streamlit_lottie import st_lottie
from nltk.tokenize import word_tokenize, regexp_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import SentenceTransformer





# Display lottie animations
def load_lottieurl(url):

    # get the url
    r = requests.get(url)
    # if error 200 raised return Nothing
    if r.status_code !=200:
        return None
    return r.json()

## ----Preprocessing Functions----

# Handle Swear Words
def replaceSwear(text):
    '''Replace censored words by the actual words'''

    text = re.sub(r'F[\*]+k', 'Fuck', text)
    text = re.sub(r'A[\*]{2}e', 'Arse', text)
    text = re.sub(r'N\*gger', 'Nigger', text)
    text = re.sub(r'nig\*\*s', 'niggers', text)
    text = re.sub(r'f\*ck', 'fuck', text)
    text = re.sub(r'F\*ck', 'Fuck', text)
    text = re.sub(r'[f][\*]+k', 'fuck', text)
    text = re.sub(r'ho\*s', 'hoes', text)
    text = re.sub(r'N\*ggas', 'Niggas', text)
    text = re.sub(r'f[\*]+ing', 'fucking', text)
    text = re.sub(r'n\*gga', 'nigga', text)
    text = re.sub(r'motherf[\*]+ker', 'motherfucker', text)
    text = re.sub(r'p[\*]{3}y', 'pussy', text)
    text = re.sub(r'N[\*]+s', 'Niggers', text)
    text = re.sub(r'B[\*]+s', 'Bitches', text)
    text = re.sub(r'Sh\*t', 'Shit', text)
    text = re.sub(r'muthaf[\*]+in', 'muthafuckin', text)
    text = re.sub(r'b!tches', 'bitches', text)
    text = re.sub(r'b!tch', 'bitch', text)
    text = re.sub(r'sh!t', 'shit', text)
    text = re.sub(r'D!ck', 'Dick', text)
    text = re.sub(r'pu\$\$y', 'pussy', text)
    text = re.sub(r'ho\%s', 'hoes', text)
    text = re.sub(r'ho\%e' , 'hoe', text)

    return text

# Handle Accented Words
def replace_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

#Create a function to clean the tweets
def cleanText(text):
    '''Cleans tweet'''

    text = re.sub(r'@[\w]+', ' ', text) #Removing @mentions
    text = re.sub(r'\&?#[\w]+;?', ' ', text) #Remove #hashtags
    text = re.sub(r'RT[\s]?:?', ' ', text) #Removing RT
    text = re.sub(r'https?:\/\/\S+', ' ', text) #Remove the hyper link
    text = re.sub(r'^[!]+', ' ', text) # Remove ! at the start
    text = re.sub(r'(\&amp\;)', ' ', text) # Remove ampersand
    text = contractions.fix(text)
    text = re.sub('\"|\'', ' ',text) # remove quotation marks
    text = re.sub(r'#', ' ', text) #Remove the '#' symbol
    text = re.sub('[^\w\s]', ' ', text) # remove punctuation
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = re.sub('[^A-Za-z0-9\s]+', '', text)
    text = re.sub('\s+', ' ', text) # if there's more than 1 whitespace, then make it just 1
    text = text.lower() # lowercase
    text = text.strip() # remove white space at the start and end of the sentence

    return text

# Fix Profanity
def fix_profanity(text):
    ''' Spell correction for profanity'''

    text = re.sub(r'[Dd]ouche\s?[Bb]ags?', 'douchebag', text)
    text = re.sub(r'ass\s?hole', 'asshole', text)
    text = re.sub(r'mother\s?fucker', 'motherfucker', text)
    text = re.sub(r'butt\shole', 'butthole', text)

    return text

# Remove any extra chracaters
def removeMoreThanOneChar(word):
    ''' Remove more than one char for a word '''

    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)') #pattern used to identify characters that occur twice among other characters in the word.
    match_substitution = r'\1\2\3'
    while (True):
        # check for semantically correct word
        if wordnet.synsets(word):
            return word
        # remove one repeated character
        new_word = repeat_pattern.sub(match_substitution, word)
        if new_word != word:
            # update old word to last substituted state
            word = new_word
        else:
               return new_word


def removeMoreThanOneChar1(text):
    ''' Remove more than one char for a sequence '''

    cleaned_tokens=[]
    token_list=word_tokenize(text)
    for token in token_list:
      token = removeMoreThanOneChar(token)
      cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


# Load slang words dictionary
with open('Abbr_dict', 'r') as file:
    Abbr_dict = json.load(file)


def replace_slang(tweet):
    tokens = nltk.word_tokenize(tweet)
    tokens = [Abbr_dict[token].lower() if token in Abbr_dict.keys() else token for token in tokens]

    return " ".join(tokens)


## Page Configuration
st.set_page_config(page_title="Tweet Hate Detector",
                   page_icon=":page_facing_up:")




st.title('Twitter Hate and Offensive Speech Detection')
col1, col2 = st.columns(2)
with col1:
    st.markdown("""

        Created by **Antoine Rahal**, **Bahige Saab** and **Rafic Srouji**.

        <div style="text-align: justify;">
        This project aims to <strong>automate content moderation</strong> to identify hate and offensive speech using <strong><strong>machine learning and deep learning multi-class classification algorithms.</strong>
        <br></br>

        Several models were tested that included Logistic Regression, Random Forest, Extreme Gradient Boosting, Support Vector Machine (SVM), K-Nearest Neighbours, Multi Perceptron Layer,
        and Long Short Term Memory. The final model was a <strong>Stacking Classifier</strong> model that used word contexualtized embeddings as features. Extreme Gradient Boosting, Support Vector Machine (SVM), K-Nearest Neighbours and
         Long Short Term Memory as base learners, and Logistic Regression as a meta-learner with accuracy 85%.
         </div>
        """,unsafe_allow_html = True)

# Defining some functions we need in the project
# def show_classification(classification):
#     if classification == "Hate":
#         return st.error("This is a hate tweet.")
#     elif classification == "Offensive":
#         return st.warning("This is an offensive tweet.")
#     elif classification == "None":
#         return st.info("This tweet is neither hate nor offensive.")

st.title("Hate Speech Detector")

st.subheader("Kindly enter a tweet to determine if it is a normal or offensive or hate tweet:")

class KerasClassifier(tf.keras.wrappers.scikit_learn.KerasClassifier):
    """
    TensorFlow Keras API neural network classifier.

    Workaround the tf.keras.wrappers.scikit_learn.KerasClassifier serialization
    issue using BytesIO and HDF5 in order to enable pickle dumps.

    Adapted from: https://github.com/keras-team/keras/issues/4274#issuecomment-519226139
    """

    def _getstate_(self):
        state = self._dict_
        if "model" in state:
            model = state["model"]
            model_hdf5_bio = io.BytesIO()
            with h5py.File(model_hdf5_bio, mode="w") as file:
                model.save(file)
            state["model"] = model_hdf5_bio
            state_copy = copy.deepcopy(state)
            state["model"] = model
            return state_copy
        else:
            return state

    def _setstate_(self, state):
        if "model" in state:
            model_hdf5_bio = state["model"]
            with h5py.File(model_hdf5_bio, mode="r") as file:
                state["model"] = tf.keras.models.load_model(file)
        self._dict_ = state


tweet = st.text_input("Enter Tweet Please: ",max_chars=280)
# st.write('The tweet to be classified is: ', tweet)
tweet = replaceSwear(tweet)
tweet = replace_accented_chars(tweet)
tweet = cleanText(tweet)
tweet = fix_profanity(tweet)
tweet = removeMoreThanOneChar1(tweet)
tweet = replace_slang(tweet)
 # loading in model
# final_model = pickle.load(open(r'D:\Downloads\svm_pkl', 'rb'))
final_model = pickle.load(open(r"xgb.pkl", 'rb'))


bert = SentenceTransformer(r"C:\Users\Rafic Srouji\MSBA 316\project\all-distilroberta-v1.bert")

predict = st.button('Predict')

if predict:
    tweet = bert.encode([tweet])
    prediction = final_model.predict(tweet)

    if prediction == 0:
        st.error("Hate Speech")
    elif prediction == 1:
        st.warning("Offensive Speech")
    else:
        st.success("Neither Hate nor Offensive")

    probabilities = final_model.predict_proba(tweet)
    probabilities_df = pd.DataFrame(probabilities, columns = ['Hate','Offensive','Neither']).T.reset_index()
    probabilities_df.columns = ['Label','Probability']
    probabilities_df = probabilities_df.sort_values('Probability',ascending=True)
    fig = px.bar(probabilities_df, x="Probability", y="Label", title="Probability Per Label",text = "Probability",text_auto =',.0%')
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False,visible=False)
    fig.update_traces(textfont_size=16, textangle=0, textposition="outside", cliponaxis=False,width=0.2)

    st.write(fig)
    # probabilities_df = pd.melt(probabilities_df, id_vars =['Name'], value_vars =['Course'])

# classification = st.text_input("Enter Classification: ")
# show_classification(classification)
