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
chroma_collection = "Tweets"
reset_collection = True
sentence_embedder_id = "sentence-transformers/all-MiniLM-L6-v2"
similarity_type = "cosine"
BATCH_SIZE = 512
LIMIT = 1000000
upstream = {
    "txt_to_parquet": {
        "nb": "/home/ubuntu/Agata/product/get/txt_to_parquet.ipynb",
        "data": "/home/ubuntu/Agata/product/get/tweets.parquet",
    }
}
product = {"nb": "/home/ubuntu/Agata/product/get/create_chroma_db.ipynb"}


# %%
# your code here...

# %% [markdown]
# # Chroma DB Creation
# Creates a database of embeddings

# %%
from torch import cuda
from sentence_transformers import SentenceTransformer
from tqdm.auto import trange, tqdm
import chromadb
import pandas as pd
import torch

# %%
df = pd.read_parquet(upstream["txt_to_parquet"]["data"]).iloc[:LIMIT]

# %%
df.head()

# %%


device = torch.device("cuda") if cuda.is_available() else torch.device("cpu")
if device.type == "cuda":
    print("Using cuda")
else:
    print(f"Not using cuda, using {device} instead")

# %%
chroma_client = chromadb.PersistentClient()

if reset_collection:
    if chroma_collection in [col.name for col in chroma_client.list_collections()]:
        chroma_client.delete_collection(chroma_collection)
        print("Deleted existing collection.")


SIMILARITY_TYPES = ["cosine", "l2", "ip"]
assert (
    similarity_type in SIMILARITY_TYPES
), f"similarity type '{similarity_type}' not supported by cromaDB. Try with on of these {SIMILARITY_TYPES}"
collection = chroma_client.get_or_create_collection(
    name=chroma_collection, metadata={"hnsw:space": similarity_type}
)

# %%
embedder_model = SentenceTransformer(
    sentence_embedder_id,
    device=device,
)

# %%
# get embedding size
docs = ["Ziopera"]
embeddings = embedder_model.encode(docs)

print(f"Dimensionality of embedding: {len(embeddings[0])}.")

# %%

metadata = [
    {
        "ID_1": int(df["ID_1"].iloc[i]),
        "ID_2": int(df["ID_2"].iloc[i]),
        "handle": df["handle"].iloc[i],
        "timestamp": df["timestamp"].iloc[i].timestamp(),
        "lenght": len(df["content"].iloc[i]),
    }
    for i in trange(len(df))
]

# %% [markdown]
# Running time of this embedder is 6:40 minutes on a dataset of roughly 2M tweets.

# %%
emb = embedder_model.encode(
    df["content"].to_list(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    device=device,
)


# %% [markdown]
# 23 Minutes fro 1 Million Samples

# %%
INSERT_BS = 10000
for i in trange(0, len(emb), INSERT_BS):
    ids = [str(id) for id in range(i, i + INSERT_BS)]
    embeddings = emb[i : i + INSERT_BS, :].tolist()
    meta = metadata[i : i + INSERT_BS]
    docsearch = collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=meta,
        documents=df["content"].iloc[i : i + INSERT_BS].to_list(),
    )