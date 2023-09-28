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
file_paths = [
    "data/Cheng-Caverlee-Lee/training_set_tweets.txt",
    "data/Cheng-Caverlee-Lee/test_set_tweets.txt",
]
product = {
    "nb": "/home/ubuntu/Agata/product/get/txt_to_parquet.ipynb",
    "data": "/home/ubuntu/Agata/product/get/tweets.parquet",
}


# %%
import re
from tqdm.auto import tqdm, trange
from datetime import datetime

# Regular expression pattern
pattern = r"^(\d+)\s+(\d+)\s+@([^\s]+)\s+([^\n]+)\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})$"


# Initialize a list to store the parsed data
parsed_data = []
for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as file:
        for line in tqdm(file):
            line = (
                line.strip()
            )  # Remove leading/trailing whitespace and newline characters
            match = re.match(pattern, line)
            if match:
                # Extract the matched groups
                integer1 = int(match.group(1))
                integer2 = int(match.group(2))
                string_without_spaces = match.group(3)
                string_with_spaces_and_punctuation = match.group(4)
                date = match.group(5)
                time = match.group(6)

                # Add the parsed data to the list
                parsed_data.append(
                    {
                        "ID_1": integer1,
                        "ID_2": integer2,
                        "handle": f"@{string_without_spaces}",
                        "content": string_with_spaces_and_punctuation,
                        "Date": date,
                        "Time": time,
                        "timestamp":datetime.strptime(date + " " + time, "%Y-%m-%d %H:%M:%S")
                    }
                )


# %%
import pandas as pd

df = pd.DataFrame(data=parsed_data)


# %%
df.to_parquet(product["data"])
