import io
from typing import List
import gradio as gr
from matplotlib.colors import Normalize
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

from lib import metrics


# tweets = [
#     "Just started learning #Python today. Excited to dive into the world of coding!",
#     "Woke up early to catch the sunrise. Nature's beauty never ceases to amaze me. 🌅 #NatureLover",
#     "Spent the evening reading a great book. There's nothing like getting lost in a good story. 📚 #Bookworm",
#     "Coding late into the night. The world of technology never sleeps! 💻 #Programming",
#     "Just had the most delicious meal at my favorite restaurant. 😋 #Foodie",
#     "Workout complete! Feeling strong and motivated. 💪 #FitnessGoals",
#     "Visited a new art exhibit today. Art has a unique way of inspiring creativity. 🎨 #ArtLover",
#     "Sundays are for relaxation and self-care. How do you unwind on the weekend? #SelfCareSunday",
#     "Hiking through the mountains today. The views are breathtaking. ⛰️ #Adventure",
#     "Cooked a homemade meal from scratch. It's all about the love you put into it. #Cooking",
#     "Just watched a thought-provoking documentary. It's essential to stay informed and open-minded. 📽️ #Documentary",
#     "Met up with old friends for a reunion. Laughter and nostalgia filled the air. 👫 #Friendship",
#     "Started a new hobby: gardening! Watching plants grow is so satisfying. 🌱 #Gardening",
#     "Grateful for the little things in life: a warm cup of tea and a cozy blanket on a rainy day. ☕🌧️ #Gratitude",
#     "Exploring a new city today. Travel broadens the mind and feeds the soul. 🌍 #Travel",
#     "Friday night movie marathon with popcorn and pajamas. The perfect way to unwind. 🍿🎬 #MovieNight",
#     "Learning a new language is both challenging and rewarding. 🌏 #LanguageLearning",
#     "Taking a break from screens to enjoy some quality time with family. 👨‍👩‍👧‍👦 #FamilyTime",
#     "Attended an inspiring conference today. Knowledge is power! 📚 #Education",
#     "Staying positive and focused on my goals. Every day is a new opportunity. 💪 #Motivation",
#     "Just adopted a rescue pet. Welcome to the family, furry friend! 🐾 #PetLove",
#     "Spent the weekend in the great outdoors, camping and stargazing. 🏕️✨ #Nature",
#     "Sometimes, all you need is a good cup of coffee to kickstart the day. ☕ #CoffeeLover",
#     "Volunteered at the local shelter today. Small acts of kindness can make a big difference. 🤝 #Volunteer",
#     "Feeling inspired by the beauty of the natural world. Let's protect our planet for future generations. 🌎 #Environment",
#     "Set a new personal record at the gym today. Hard work pays off! 💪 #Fitness",
#     "Cooked up a storm in the kitchen and invited friends over for a feast. Good food, good company. 🍽️ #Foodie",
#     "Took a spontaneous road trip with no destination in mind. Sometimes, the best adventures are unplanned. 🚗 #Adventure",
#     "Rediscovered the joy of reading. Books have the power to transport us to different worlds. 📖 #Reading",
#     "Spent the day at the beach, soaking up the sun and listening to the waves. 🏖️ #BeachDay",
# ]

# Parameters
chroma_collection = "Tweets"
reset_collection = True
sentence_embedder_id = "sentence-transformers/all-MiniLM-L6-v2"
similarity_type = "cosine"


global_topic_list: List[str] = []
returned_tweets: List[str] = []
returned_tweets_emb: List[np.array] = []

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
sentiment_classifier = pipeline("sentiment-analysis")

chroma_client = chromadb.PersistentClient()
chroma_collection = "Tweets"
collection: chromadb.Collection = chroma_client.get_collection(
    name=chroma_collection,
    embedding_function=embedder_model,
)


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
    return sentiment_classifier(tweets)


def search(query, k, min_len, topics_string) -> List[str]:
    global collection
    global embedder_model
    global global_topic_list
    global returned_tweets
    global returned_tweets_emb

    k = int(k)
    print("Querying Database...")
    start = time.time()
    response = collection.query(
        query_embeddings=embedder_model.encode(query).tolist(),
        n_results=N_TWEETS,
        where={
            "lenght": {"$gte": int(min_len)},
        },
    )
    print(f"Execution Time:{(time.time()-start):.2f}\n")
    # sentiments analysis
    tweets = response["documents"][0]
    tweets_embs = response["embeddings"]
    # update global variables for plot
    returned_tweets = tweets[:]
    returned_tweets_emb = tweets_embs
    # reduce to k the lists
    tweets = tweets[0:k]

    print("Analysing Sentiment...")
    start = time.time()
    sentiments = sentiment_analysis(tweets)
    print(f"Execution Time:{(time.time()-start):.2f}\n")
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
        embedded_tweets = embedder_model.encode(tweets).tolist()
        topic_query_result = topic_collection.query(embedded_tweets, n_results=1)[
            "documents"
        ]
        ram_chroma_client.delete_collection(TOPIC_COLLECTION_NAME)
        print(f"Execution Time:{(time.time()-start):.2f}\n")
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
    global returned_tweets
    global returned_tweets_emb

    if not topic in global_topic_list:
        print("Warning! given topic not in list!!!")

    encoded_topic = embedder_model.encode(topic)
    embedded_tweets = embedder_model.encode(returned_tweets)

    similarities = metrics.cosine_similarity(
        encoded_topic, np.transpose(embedded_tweets)
    )
    pca = PCA(n_components=PCA_DIMENSION)
    data_pca = pca.fit_transform(embedded_tweets)
    # Apply t-SNE for visualization
    data_tsne = TSNE(n_components=TSNE_DIMENSION).fit_transform(data_pca)
    # Plot with Seaborn
    # Plot with Seaborn
    fig, ax = plt.subplots(figsize=(10, 8))

    # Using a scatter plot from matplotlib to have a continuous color bar
    norm = Normalize(vmin=min(similarities), vmax=max(similarities))
    sc = ax.scatter(
        data_tsne[:, 0], data_tsne[:, 1], c=similarities, cmap="viridis", norm=norm
    )

    ax.set_title("t-SNE visualization colored by similarity to the first point")

    # Display color bar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Similarity", rotation=270, labelpad=15)

    # Convert plot to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    image.load()
    buf.close()

    return image


def generate_plot_from_features(topic: str) -> Image.Image:
    global global_topic_list
    global collection

    if not topic in global_topic_list:
        print("Warning! given topic not in list!!!")

    embeddings = collection.get()
    print(len(embeddings))
    return None


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
            "# Grapho Search\n"
            "Enter a question, and get a list of similar tweets from the database."
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
