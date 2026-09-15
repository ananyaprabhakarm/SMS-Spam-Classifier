import streamlit as st
import pickle
import string
import pandas as pd
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

for resource in ('punkt_tab', 'stopwords'):
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt_tab' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

ps = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in STOP_WORDS and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

HAM_IDX = list(model.classes_).index(0)
SPAM_IDX = list(model.classes_).index(1)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message.")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        spam_probability = model.predict_proba(vector_input)[0][SPAM_IDX]

        # 4. Display
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")

        st.metric("Confidence it's spam", f"{spam_probability:.1%}")
        st.progress(spam_probability)

        with st.expander("How the model got here"):
            st.caption(
                "Cleaned text the model actually sees "
                "(lowercased, stopwords/punctuation removed, stemmed):"
            )
            st.code(transformed_sms or "(nothing left after cleaning)")

            vocabulary = tfidf.vocabulary_
            log_prob_ham = model.feature_log_prob_[HAM_IDX]
            log_prob_spam = model.feature_log_prob_[SPAM_IDX]

            contributions = []
            for word in dict.fromkeys(transformed_sms.split()):
                idx = vocabulary.get(word)
                if idx is not None:
                    contributions.append((word, log_prob_spam[idx] - log_prob_ham[idx]))

            if contributions:
                contributions.sort(key=lambda w: abs(w[1]), reverse=True)
                st.caption(
                    "Words that most pushed this prediction toward spam (+) "
                    "or not-spam (−), based on how strongly the model "
                    "associates each word with each class:"
                )
                chart_df = pd.DataFrame(
                    contributions[:10], columns=["word", "spam lean"]
                ).set_index("word")
                st.bar_chart(chart_df)
            else:
                st.caption(
                    "None of the cleaned words were seen during training, "
                    "so this prediction relies on the model's baseline "
                    "class probabilities rather than any specific word."
                )
