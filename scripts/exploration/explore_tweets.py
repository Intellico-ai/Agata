# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: ploomber
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   ploomber:
#     injected_manually: true
# ---

# %% tags=["parameters"]
# add default values for parameters here

# %% tags=["injected-parameters"]
# Parameters
upstream = {
    "txt_to_parquet": {
        "nb": "/home/ubuntu/Agata/product/get/txt_to_parquet.ipynb",
        "data": "/home/ubuntu/Agata/product/get/tweets.parquet",
    }
}
product = {"nb": "/home/ubuntu/Agata/product/exploration/explore_tweets.ipynb"}


# %% [markdown]
# # Data Exploration
# The objective of this data exploration is to understand the distribution of the wqords in the dataset.


# %%
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from tqdm.auto import tqdm, trange
from collections import Counter
from nltk.text import Text
import pylab
import spacy
from nltk.tokenize.toktok import ToktokTokenizer
from spacy import displacy

# %%

LIMIT = 10000
if LIMIT is not None:
    df = pd.read_parquet(upstream["txt_to_parquet"]["data"]).sample(LIMIT)
else:
    df = pd.read_parquet(upstream["txt_to_parquet"]["data"])

# %%
len(df)

# %% [markdown]
# ## Data Cleaning
# Let's lowercase every word, remove urls, remove mentions, special characters and so on.

# %%
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# %%
import unicodedata
from lib.contractions import CONTRACTION_MAP


def preprocess_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    # Remove user mentions and '#' from tweet
    tweet = re.sub(r"\@\w+|\#", "", tweet)
    # Remove punctuations
    tweet = re.sub(r"\W", " ", tweet)
    # Remove digits
    # tweet = re.sub(r'\d+', '', tweet)
    # Remove stopwords
    tweet_tokens = nltk.word_tokenize(tweet)
    filtered_words = [word for word in tweet_tokens if word not in stop_words]

    return filtered_words


def remove_accented_chars(text):
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = (
            contraction_mapping.get(match)
            if contraction_mapping.get(match)
            else contraction_mapping.get(match.lower())
        )
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = " ".join([ps.stem(word) for word in text.split()])
    return text


def lemmatize_text(text, nlp):
    text = nlp(text)
    text = " ".join(
        [word.lemma_ if word.lemma_ != "-PRON-" else word.text for word in text]
    )
    return text


def remove_stopwords(text, tokenizer, stopword_list, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [
            token for token in tokens if token.lower() not in stopword_list
        ]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text


# %%

nlp = spacy.load("en_core_web_md")
# nlp_vec = spacy.load('en_vecs', parse = True, tag=True, #entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words("english")
stopword_list.remove("no")
stopword_list.remove("not")

# %%
tweets = df["content"].tolist()
# lower so no problems with character
tweets = [tweet.lower() for tweet in tqdm(tweets, desc="lowering")]
# remove accented characters just to be safe
tweets = [
    remove_accented_chars(tweet) for tweet in tqdm(tweets, desc="removing accents")
]
tweets = [
    re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    for tweet in tqdm(tweets, desc="Removing Links")
]
tweets = [
    re.sub(r"\W", " ", tweet) for tweet in tqdm(tweets, desc="Removing Punctuation")
]
# expand contractions so it's more interpetable
tweets = [
    expand_contractions(tweet) for tweet in tqdm(tweets, desc="expanding contractions")
]
# find root of words
tweets = [lemmatize_text(tweet, nlp) for tweet in tqdm(tweets, desc="lemmization")]
# remove stopwords
tweets = [
    remove_stopwords(tweet, tokenizer, stopword_list)
    for tweet in tqdm(tweets, desc="remove stopwords")
]
len(tweets)


# %%
embedded = []
for tweet in tqdm(tweets):
    tweet_emb = nlp(tweet)
    embedded.append(tweet_emb)

# %%
flattened_list = [item for sublist in embedded for item in sublist]

# %%


# %%
nouns_list = [
    word.text
    for word in tqdm(flattened_list, desc="Extracting Nouns")
    if word.pos_ == "NOUN"
]
adj_list = [
    word.text
    for word in tqdm(flattened_list, desc="Extracting Nouns")
    if word.pos_ == "ADJ"
]
verb_list = [
    word.text
    for word in tqdm(flattened_list, desc="Extracting Nouns")
    if word.pos_ == "VERB"
]


# %%
len(nouns_list)

# %%
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np


def word_dataframe(word_embs):
    text = [word.text for word in word_embs]
    vectors = [word.vector for word in word_embs]
    return pd.DataFrame({"text": text, "embedding": vectors})


def dimensionality_reduction(word_df):
    # Reduce dimensions to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    word_vectors_2d = tsne.fit_transform(np.stack(word_df["embedding"]))
    return word_vectors_2d


def generate_word_cloud(data, title, use_tsne=False):
    # Combine lists of words into a single string
    text = " ".join(data)

    # Create the word cloud object
    wc = WordCloud(
        width=600,
        height=600,
        background_color="white",
        colormap="tab20c",
        max_words=1000,
    ).generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(title)
    plt.axis("off")
    plt.show()
    return wc


# Generate word clouds for each category
_ = generate_word_cloud(nouns_list, "Nouns Word Cloud")
_ = generate_word_cloud(adj_list, "Adjectives Word Cloud")
wc = generate_word_cloud(verb_list, "Verbs Word Cloud")


# %%
embedded_tweets = embedded
tokenized_tweets = [[token.text for token in doc] for doc in embedded_tweets]

# %%
from gensim import corpora

# Create a Gensim Dictionary
dictionary = corpora.Dictionary(tokenized_tweets)

# Create a Gensim Corpus
corpus = [dictionary.doc2bow(tweet) for tweet in tokenized_tweets]
from gensim.models import LdaModel

# Choose the number of topics you want to extract
num_topics = 5  # You can adjust this number based on your requirements

# Create and train the LDA model
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Print the topics and their top words
for topic_id, topic_words in lda_model.print_topics():
    print(f"Topic {topic_id + 1}: {topic_words}\n")


# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Create an instance of the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.85, max_features=2000
)  # You can adjust parameters as needed

# Fit and transform your preprocessed tweets to obtain the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_tweets)

# The resulting tfidf_matrix is a sparse matrix that contains the TF-IDF values


# %%
from sklearn.decomposition import NMF

# Create the term-document matrix (you can use TF-IDF or other vectorization methods)
# tfidf_matrix is assumed to be your term-document matrix
# You should have a proper vectorization of your text data here
# For example, you can use TfidfVectorizer from scikit-learn

# Choose the number of topics (components) you want to extract
num_topics = 5  # You can adjust this number based on your requirements

# Apply NMF
nmf = NMF(n_components=num_topics)
nmf.fit(tfidf_matrix)

# Get the topics and their top words
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(nmf.components_):
    print(f"Topic {topic_idx + 1}:")
    top_words_idx = topic.argsort()[-10:][::-1]  # Get the indices of the top 10 words
    top_words = [feature_names[i] for i in top_words_idx]
    print(" ".join(top_words))
