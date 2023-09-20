# %%
# Parameters
file_paths = [
    "data/Cheng-Caverlee-Lee/training_set_tweets.txt",
    "data/Cheng-Caverlee-Lee/test_set_tweets.txt",
]
product = {
    "nb": "/home/ubuntu/NGI-Search-Demo/product/get/txt_to_parquet.ipynb",
    "data": "/home/ubuntu/NGI-Search-Demo/product/get/tweets.parquet",
}


# %%
import re
from tqdm.auto import tqdm, trange

# Regular expression pattern
pattern = r'^(\d+)\s+(\d+)\s+@([^\s]+)\s+([^\n]+)\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})$'



# Initialize a list to store the parsed data
parsed_data = []
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            line = line.strip()  # Remove leading/trailing whitespace and newline characters
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
                parsed_data.append({
                    "ID_1": integer1,
                    "ID_2": integer2,
                    "handle": f"@{string_without_spaces}",
                    "content": string_with_spaces_and_punctuation,
                    "Date": date,
                    "Time": time
                })


# %%
import pandas as pd
df = pd.DataFrame(data=parsed_data)


# %%
df.to_parquet(product["data"])


