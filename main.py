import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
import plotly.express as px
from collections import Counter
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Chargement des données
@st.cache(allow_output_mutation=True)
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Sentiment'] = df['Sentiment'].replace({0: 'Negative', 1: 'Positive'})
    df['Tweet'] = df['Tweet'].apply(preprocess_text)
    return df

def get_sentiment_prediction(tweet_text):
    tweet_text = preprocess_text(tweet_text)
    
    url = "https://6310-2a01-cb04-ae6-c900-d6a1-afc7-bf95-2370.ngrok-free.app/predict-sentiment/"
    payload = {"text": tweet_text}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Une erreur s'est produite lors de la communication avec l'API de prédiction.")
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")

# Fonctions de prétraitement du texte
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_spaces(text)
    text = remove_retweets(text)
    text = remove_stopwords_and_short_words(text)
    text = to_lowercase(text)
    text = convert_emojis_to_text(text)
    return text

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_mentions(text):
    return re.sub(r'@[A-Za-z0-9_]+', '', text)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_extra_spaces(text):
    return re.sub(' +', ' ', text.strip())

def remove_retweets(text):
    return re.sub(r'\brt\b', '', text)

def remove_stopwords_and_short_words(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words and len(word) > 1])

def to_lowercase(text):
    return text.lower()

def convert_emojis_to_text(text):
    emoji_dictionary = {
        "🙂": ":)",
        "😊": ":)",
        "😁": ":D",
        "😂": ":D",
        "😢": ":(",
        "😭": ":'(",
        "😠": ">:(",
        "😡": ">:(",
        "😍": "<3",
        "😘": ":*",
        "😎": "8-)",
        "🤔": ":think:",
        "😜": ";P",
        "🤗": "hug",
        "🙁": ":(",
        "🤨": ":|",
        "🤓": "nerd",
        "😩": ":o",
        "🤤": "drool",
        "👍": "thumbs_up",
        "👎": "thumbs_down",
        "❤️": "heart",
        "🙌": "raise_the_roof",
        "😏": "smirk",
        "😒": "unamused",
        "🤩": "star_eyes",
        "🤪": "crazy",
        "🤬": "angry_words",
        "😇": "angel"
    }

    for emoji, replacement in emoji_dictionary.items():
        text = text.replace(emoji, replacement)

    return text

# Calcul de la taille des phrases
def compute_sentence_lengths(data):
    data['sentence_length'] = data['Tweet'].apply(lambda x: len(x.split()))
    return data

# Calcul de la fréquence des mots (10 mots les plus fréquents)
def compute_word_frequencies(data, sentiment):
    sentiment_data = data[data['Sentiment'] == sentiment]
    all_words = ' '.join(sentiment_data['Tweet']).split()
    most_common_words = Counter(all_words).most_common(10)
    words = [word[0] for word in most_common_words]
    counts = [word[1] for word in most_common_words]
    return words, counts

# Fonction pour générer un nuage de mots
def generate_wordcloud(text, title, bg_color):
    wordcloud = WordCloud(background_color=bg_color, max_words=7).generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))  # Création de la figure et des axes
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)  

# Fonction principale pour exécuter l'application Streamlit
def main():
    st.title("Dashboard d'Analyse de Sentiment")

    # Chargement et affichage des données
    data = load_data('df.csv')
    st.write(data.head())

    # Choix du type de graphique pour la distribution des sentiments
    chart_type = st.sidebar.selectbox("Choisissez le type de graphique pour la distribution des sentiments:", ["Barplot", "Pie Chart"])

    # Distribution des sentiments
    st.subheader('Distribution des Sentiments')
    sentiment_count = data['Sentiment'].value_counts()
    sentiment_count_df = pd.DataFrame({'Sentiment': sentiment_count.index, 'Count': sentiment_count.values})
    
    if chart_type == "Barplot":
        fig = px.bar(sentiment_count_df, x='Sentiment', y='Count', color='Sentiment', title="Distribution des Sentiments")
        st.plotly_chart(fig)
    elif chart_type == "Pie Chart":
        fig = px.pie(sentiment_count_df, names='Sentiment', values='Count', title="Distribution des Sentiments")
        st.plotly_chart(fig)

    # Analyse statistique des Tweets
    data_with_length = compute_sentence_lengths(data)
    words, counts = compute_word_frequencies(data_with_length, 'Positive')

    st.subheader('Analyse Statistique des Tweets')
    fig = px.histogram(data_with_length, x='sentence_length', nbins=20, title='Distribution de la Taille des Phrases')
    st.plotly_chart(fig)

    fig = px.bar(x=words, y=counts, labels={'x':'Mot', 'y':'Fréquence'}, title='Fréquence des 10 Mots les Plus Fréquents (Positif)')
    st.plotly_chart(fig)

    words, counts = compute_word_frequencies(data_with_length, 'Negative')

    fig = px.bar(x=words, y=counts, labels={'x':'Mot', 'y':'Fréquence'}, title='Fréquence des 10 Mots les Plus Fréquents (Négatif)')
    st.plotly_chart(fig)

    # WordClouds et prédiction de sentiment...
    # WordCloud pour tweets positifs
    st.subheader('WordCloud pour les Tweets Positifs')
    positive_data = data[data['Sentiment'] == 'Positive']['Tweet']
    positive_text = ' '.join(positive_data)
    generate_wordcloud(positive_text, "WordCloud pour les Tweets Positifs", "white")

    # WordCloud pour tweets négatifs
    st.subheader('WordCloud pour les Tweets Négatifs')
    negative_data = data[data['Sentiment'] == 'Negative']['Tweet']
    negative_text = ' '.join(negative_data)
    generate_wordcloud(negative_text, "WordCloud pour les Tweets Négatifs", "black")

    # Section de prédiction de sentiment d'un nouveau tweet
    st.subheader("Prédiction de Sentiment d'un Nouveau Tweet")
    tweet_text = st.text_area("Entrez le tweet à analyser", "")
    if st.button("Analyser le sentiment"):
        prediction = get_sentiment_prediction(tweet_text)
        if prediction:
            # Affiche uniquement le sentiment prédit
            st.write(f"Sentiment prédit : {prediction['sentiment']}")

if __name__ == "__main__":
    main()
