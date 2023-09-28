from datetime import datetime
import io
from typing import List, Dict, Optional
import gradio as gr
from matplotlib.colors import Normalize
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.ticker import MaxNLocator
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import trange, tqdm
import chromadb
import pandas as pd
import torch
import time
from transformers import pipeline
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src import metrics
from src import plot
from src.app.structures import SearchResult


# Parameters
chroma_collection = "Tweets"
reset_collection = True
sentence_embedder_id = "sentence-transformers/all-MiniLM-L6-v2"
similarity_type = "cosine"


global_topic_list: List[str] = []
search_result: Optional[SearchResult] = None

MAX_TEXT_BOXES = 100
USE_STANDARD_INTERFACE = False
PCA_DIMENSION = 10
TSNE_DIMENSION = 2
N_TWEETS = 2000
USE_TSNE = False


# instantiate model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device.type == "cuda":
    print("Using cuda")
else:
    print(f"Not using cuda, using {device} instead")

embedder_model = SentenceTransformer(
    sentence_embedder_id,
    device=device,
)


# create pipeline for sentiment analysis
sentiment_classifier = pipeline("sentiment-analysis", device=device)

chroma_client = chromadb.PersistentClient()
chroma_collection = "Tweets"
collection: chromadb.Collection = chroma_client.get_collection(
    name=chroma_collection,
    embedding_function=embedder_model,
)


def get_or_create_global_embedding_list():
    """Factory method for creating ondemand the whole embedding list

    Returns:
        _type_: _description_
    """
    global all_db
    if all_db == {}:
        all_db = collection.get(include=["embeddings", "metadatas"])
    return all_db


def variable_outputs(k):
    global MAX_TEXT_BOXES
    k = int(k)
    return [gr.Textbox.update(visible=True)] * k + [
        gr.Textbox.update(visible=False)
    ] * (MAX_TEXT_BOXES - k)


def parse_topics(topic_string: str):
    """Converts the string given by the client into a list of topics.
    Topics are divided with commands. Leading and Trailing spaces are removed

    Args:
        topic_string (_type_): _description_

    Returns:
        _type_: _description_
    """
    topics_list = [topic.strip() for topic in topic_string.split(",")]
    return topics_list


def sentiment_analysis(tweets: List[str]):
    global sentiment_classifier
    return sentiment_classifier(tweets, batch_size=256)


def search(query, k, min_len, topics_string) -> List[str]:
    global collection
    global embedder_model
    global global_topic_list
    global search_result

    k = int(k)
    print("Querying Database...")
    start = time.time()
    response = collection.query(
        query_embeddings=embedder_model.encode(query).tolist(),
        n_results=N_TWEETS,
        where={
            "lenght": {"$gte": int(min_len)},
        },
        include=["metadatas", "documents", "distances", "embeddings"],
    )
    print(f"Execution Time:{(time.time()-start):.2f}")
    # sentiments analysis
    tweets = response["documents"][0]
    tweets_embs = response["embeddings"][0]
    metadatas = response["metadatas"][0]
    search_result = SearchResult(
        texts=tweets, embeddings=tweets_embs, metadatas=metadatas
    )
    # reduce to k the lists
    tweets = tweets[0:k]
    tweets_embs = tweets_embs[0:k]

    print("Analysing Sentiment...")
    start = time.time()
    sentiments = sentiment_analysis(tweets)
    print(f"Execution Time:{(time.time()-start):.2f}")
    # topic modeling
    topic_query_result = None
    if topics_string is None or topics_string != "":
        print("Topic Modeling")
        start = time.time()
        topics: List[str] = parse_topics(topics_string)
        global_topic_list = topics
        # create a db for that
        TOPIC_COLLECTION_NAME = "Topics"
        ram_chroma_client = chromadb.Client()
        topic_collection = ram_chroma_client.create_collection(
            TOPIC_COLLECTION_NAME, embedding_function=embedder_model
        )
        topic_embeddings = embedder_model.encode(topics)
        topic_collection.add(
            ids=[str(i) for i in range(len(topics))],
            embeddings=topic_embeddings.tolist(),
            documents=topics,
        )
        topic_query_result = topic_collection.query(tweets_embs, n_results=1)[
            "documents"
        ]
        ram_chroma_client.delete_collection(TOPIC_COLLECTION_NAME)
        print(f"Execution Time:{(time.time()-start):.2f}")
    print("Output Creation...")
    # sentiment output
    for sentiment in sentiments:
        if sentiment["label"] == "POSITIVE":
            sentiment["emoji"] = "😀"
        else:
            sentiment["emoji"] = "😰"
    # topic output
    if topic_query_result is None:
        output = [
            f"{tweets[i]}\nSENTIMENT - {sentiments[i]['emoji']} with score {sentiments[i]['score']:.2f}"
            for i in range(len(tweets))
        ]
    else:
        topic_query_result = [f"#{topic[0]}" for topic in topic_query_result]
        output = [
            f"{tweets[i]}\nSENTIMENT - {sentiments[i]['emoji']} with score {sentiments[i]['score']:.2f}\n"
            f"{topic_query_result[i]}"
            for i in range(len(tweets))
        ]

    return "\n\n".join(output)
    # return response["documents"][0]


def generate_plot_tsne(topic: str) -> Image.Image:
    global global_topic_list
    global embedder_model
    global search_result

    if not topic in global_topic_list:
        print("Warning! given topic not in list!!!")

    encoded_topic = embedder_model.encode(topic)
    tweets: List[str] = search_result.texts
    tweets_embds: List[np.array] = search_result.embeddings
    metadatas: List[Dict] = search_result.metadatas

    similarities = metrics.cosine_similarity(encoded_topic, np.transpose(tweets_embds))
    pca = PCA(n_components=PCA_DIMENSION)
    data_pca = pca.fit_transform(tweets_embds)
    # Apply t-SNE for visualization
    data_tsne = TSNE(n_components=TSNE_DIMENSION).fit_transform(data_pca)
    # Plot with Seaborn
    # Plot with Seaborn
    fig, ax = plt.subplots(figsize=(10, 8))

    # Using a scatter plot from matplotlib to have a continuous color bar
    norm = Normalize(vmin=min(similarities), vmax=max(similarities))
    sc = ax.scatter(
        data_tsne[:, 0],
        data_tsne[:, 1],
        c=similarities,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        norm=norm,
    )

    ax.set_title("t-SNE visualization colored by similarity to the first point")

    # Display color bar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Similarity", rotation=270, labelpad=15)

    # Convert plot to PIL Image
    image = plot.figure_to_PIL(fig)
    return image


def generate_plot_from_features(topic: str) -> Image.Image:
    global global_topic_list
    global search_result

    if not topic in global_topic_list:
        print("Warning! given topic not in list!!!")

    encoded_topic = embedder_model.encode(topic)
    tweets: List[str] = search_result.texts
    tweets_embds: List[np.array] = search_result.embeddings
    metadatas: List[Dict] = search_result.metadatas

    print("Computing similarities with topic...")
    start = time.time()
    similarities = metrics.cosine_similarity(encoded_topic, np.transpose(tweets_embds))
    print(f"Execution Time:{(time.time()-start):.2f}")

    x_feature = [meta["timestamp"] for meta in metadatas]
    print("Sentiment Analysis...")
    start = time.time()
    sentiments = sentiment_analysis(tweets)
    print(f"Execution Time:{(time.time()-start):.2f}")
    y_feature = []
    for sentiment in sentiments:
        if sentiment["label"] == "POSITIVE":
            y_feature.append(sentiment["score"])
        else:
            y_feature.append(-sentiment["score"])

    # Using a scatter plot from matplotlib to have a continuous color bar
    fig, ax = plt.subplots(figsize=(10, 8))
    norm = Normalize(vmin=min(similarities), vmax=max(similarities))
    sc = ax.scatter(
        x_feature,
        y_feature,
        c=similarities,
        cmap=sns.color_palette("light:b", as_cmap=True),
        norm=norm,
    )

    ax.set_title("Topic Distribution With Respect To Sentiment And Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Sentiment")
    # Display color bar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Similarity", rotation=270, labelpad=15)

    # Convert plot to PIL Image
    image = plot.figure_to_PIL(fig)
    return image


if USE_STANDARD_INTERFACE:
    n_tweet_component = gr.Number(value=20, default=20, label="Number of Tweets")
    minimum_lenght = gr.Number(
        value=120, default=120, label="Minimum number of characters"
    )
    demo = gr.Interface(
        fn=search,
        inputs=[
            gr.Textbox(lines=1, placeholder="Keywords"),
            n_tweet_component,
            minimum_lenght,
            gr.Textbox(
                lines=1,
                value="Rock, Metal, EDM, Dance, Heavy Metal, Techno, Funky, Grunge, Pop, Kpop, Rap",
            ),
        ],
        outputs=[gr.Textbox(placeholder="List of Tweets"), "image"],
        live=False,  # Disable live updates
        title="Grapho Search",
        description="Enter a question, and get a list of similar tweets from the database.",
        button="Submit",  # Add a submit button
    )
    demo.launch()
else:
    with gr.Blocks() as demo:
        # title
        gr.Markdown(
            "# Agata\n"
            "🎨 Artistic Graphical Assistant for Text Analysis 🎭\n\n"
            "Enter a question or a list of keywords. Agata will find the most relevant tweets! "
            "You can also define a list of topics and Agata will assing them to each tweet.\n\n"
            "Once you get your result, you can plot your results. You can choose a keyword and visualize"
            "which are the most relevant tweets with respect to that particular keyword.\n\n"
            "The tweets are disposed on two axis: sentiment (computed by Agata) and time."
        )
        with gr.Row():
            with gr.Column():
                keywords_box = gr.Textbox(value="Music", lines=1, label="Keywords")
                n_tweet_component = gr.Number(
                    value=100, default=20, label="Number of Tweets"
                )
                minimum_lenght = gr.Number(
                    value=120, default=120, label="Minimum number of characters"
                )
                topics_box = gr.Textbox(
                    lines=1,
                    label="List of Topics",
                    value="Rock, Metal, EDM, Dance, Heavy Metal, Techno, Funky, Grunge, Pop, Kpop, Rap",
                )
                with gr.Row():
                    clear_button = gr.ClearButton(
                        [keywords_box, n_tweet_component, minimum_lenght, topics_box]
                    )
                    search_button: gr.Button = gr.Button("Search", variant="primary")
            with gr.Column():
                tweets = gr.Textbox(placeholder="List of Tweets")
        with gr.Row():
            with gr.Column():
                dropdown_topic = gr.Textbox(lines=1, placeholder="Keyword")
                plot_button = gr.Button("Plot", variant="primary")
            with gr.Column():
                plot_image = gr.Image(type="pil")

        search_button.click(
            search,
            inputs=[keywords_box, n_tweet_component, minimum_lenght, topics_box],
            outputs=[tweets],
        )
        if USE_TSNE:
            plot_button.click(
                generate_plot_tsne, inputs=dropdown_topic, outputs=plot_image
            )
        else:
            plot_button.click(
                generate_plot_from_features, inputs=dropdown_topic, outputs=plot_image
            )
        # react to clicks on search

    demo.launch()
