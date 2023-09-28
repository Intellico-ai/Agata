
# Agata

Enhance data analytics for artist communities by implementing an engine able to handle information from trustworthy and profiled sources (e.g. social media authorized APIs), perform search by keywords and combination of keywords (e.g.: rock music and melancholy) and other filters and return:
- A rank of most relevant comments/contents each marked by keywords conveying different meaning (art domain, cultural stream, ..)
- Performs a sentiment analysis

## Punti di Forza
- Open Source
- Easy to Use
- Extendible
- Explainable

## Try our Demo!

### Run the Demo
Install the dependencies of the repository. Dependencies are handled with conda. Refer to [this site](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install conda.
```
conda env create -f environment.yml
```


If you want to run the demo, just
```
bash demo.sh
```
This command will download the dataset (roughly 3 GB of data) and launch the application. When the application is ready it will give you a link, just click. Otherwise, [click here](http://127.0.0.1:7860).

If you are interested in building the whole application, run the following commands:
```

wget https://archive.org/download/twitter_cikm_2010/twitter_cikm_2010.zip
unzip twitter_cikm_2010.zip -d data/Cheng-Caverlee-Lee
rm twitter_cikm_2010.zip
ploomber inject
ploomber build
python app.py
```
This will download a much smaller version of the dataset (~400 MB), but it will be preprocessed in order to get the final dataset.

### Demo Dataset
For the demo we are using the [Cheng-Caverlee-Lee Tweets dataset](https://archive.org/details/twitter_cikm_2010), a collection of scraped public twitter updates used in coordination with an academic project to study the geolocation data related to twittering. 