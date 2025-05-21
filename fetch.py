# Code for fetching URL documents and processing the contents of the documents
# This code is for reference only and does not need to be run because the parsed document content is already provided with the project.

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from googlesearch import search
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from langdetect import detect

# Download stopwords and tokenizer data if not already downloaded
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

# extract content of a URL
def extract_url_content(url):
    # send GET request to fetch page content
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the page content using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        text = " ".join(p.text for p in soup.find_all("p"))
        return text
    else:
        print(f"Failed to fetch page, status code: {response.status_code}")


# Process text
def process_text_content(text):
    text = text.lower()
    
    # tokenize words
    words = word_tokenize(text)

    # initialize stopwords and stemmer
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    # remove stopwords and apply stemming
    processed_words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]

    processed_text = " ".join(processed_words)
    return processed_text

# read in NASA data
nasa_data = pd.read_csv("./session_data/NASA_query-and-click.csv", header=None, names=["worker_id", "query_id", "query_term", "query_time", "click_time", "serp_rank", "page_url", "page_title", "page_description"])


# filter out english queries
KNOWN_ENG_WORDS = ["NASA", "nasa"]

def filter_english(sentences):
    english_sentences = []
    for sentence in sentences:
        if sentence in KNOWN_ENG_WORDS:
            english_sentences.append(sentence)
        else:
            try:
                if detect(sentence) == "en":  # Keep only English sentences
                    english_sentences.append(sentence)
            except:
                pass  # Skip errors
    return english_sentences

# get list of all queries for the topic
all_queries = nasa_data["query_term"].tolist()

# remove duplicate queries
all_queries = list(dict.fromkeys(all_queries))

# remove non english queries
english_queries = filter_english(all_queries)


# query using googlesearch
num_results = 30
serp_results = {}

def fetch_serps(queries, serp_results, num_results):
    try:
        for query in queries:
            print(f" Searching: {query}")
            
            try:
                results = list(search(query, num_results=num_results))
                serp_results[query] = results
                time.sleep(2)

            except Exception as e:
                print(f"Error fetching results for '{query}': {e}")
                break
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping queries due to manual interruption.")

fetch_serps(english_queries, [], num_results)

# Save results to a JSON file
with open("nasa_search_results.json", "w", encoding="utf-8") as file:
    json.dump(serp_results, file, indent=4)


# read serps for each query from the JSON file
with open("nasa_search_results.json", "r", encoding="utf-8") as file:
    cached_serps = json.load(file)

# load, process, and store the content of each document
urls_contents = {}
n = 0

try:
    for query, urls in cached_serps.items():
        n = n + 1
        urls_contents[query] = []
        
        print(n, f"Getting content for urls of {query}")
        
        for url in urls:
            print(f" Fetching: {url}")
            
            try:
                # send GET request to fetch page content
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                # Parse the page content using BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")

                text_content = " ".join(p.text for p in soup.find_all("p"))

                processed_text = process_text_content(text_content)
                
                urls_contents[query].append({url: processed_text})
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Error fetching {url}: {e}")
                
except KeyboardInterrupt:
    print("\n🛑 Manually stopped. Saving progress.")

# Save contents to a JSON file
with open("nasa_results_urls_content.json", "w", encoding="utf-8") as file:
    json.dump(urls_contents, file, indent=4)


# read content of each url
with open("nasa_results_urls_content.json", "r", encoding="utf-8") as file:
    urls_content_by_query = json.load(file)

# compute tf-idf vectors of documents per query
# Store results
tfidf_results = {}

# Process each query separately
for query, urls in urls_content_by_query.items():
    documents = []  # List to store content under the same query
    url_list = []   # List to store corresponding URLs

    # Extract content from each URL
    for entry in urls:
        for url, content in entry.items():
            documents.append(content)
            url_list.append(url)

    if not documents:  # Skip empty queries
        continue

    # Compute TF-IDF for this query's documents
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, max_features=500)
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Store TF-IDF scores along with URLs
    query_tfidf = {}
    for i, doc_vector in enumerate(tfidf_matrix):
        # tfidf_scores = dict(zip(feature_names, doc_vector.toarray()[0]))
        query_tfidf[url_list[i]] = list(doc_vector.toarray()[0])
        # query_tfidf.append({"url": url_list[i], "tfidf": tfidf_scores})

    # Save results under query
    tfidf_results[query] = query_tfidf

# Save TF-IDF results to JSON
with open("tfidf_results_by_query.json", "w", encoding="utf-8") as file:
    json.dump(tfidf_results, file, indent=4)

print("✅ TF-IDF scores computed and saved per query, including URLs!")


# fetch content of the documents visited by the users
headers=["worker_id", "query_id", "query_term", "query_time", "click_time", "serp_rank", "page_url", "page_title", "page_description"]

fetched = []
not_fetched = []

with open("./session_data/NASA_query-and-click.csv", mode="r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file, fieldnames=headers)
    
    for row in reader:
        query_term = row["query_term"]
        page_url = row["page_url"]
        
        if page_url is not None:
            did_fetch = False
            try:
                query_urls = urls_contents[query_term]

                # need to fetch content for this url
                if not any(page_url in obj for obj in query_urls):
                    print(f"{page_url} content not fetched!")
                    print(f" Fetching: {page_url} for query_term {query_term}")
                    
                    try:
                        # send GET request to fetch page content
                        response = requests.get(page_url, timeout=10)
                        response.raise_for_status()
            
                        # Parse the page content using BeautifulSoup
                        soup = BeautifulSoup(response.text, "html.parser")
            
                        text_content = " ".join(p.text for p in soup.find_all("p"))
            
                        processed_text = process_text_content(text_content)
                            
                        urls_contents[query_term].append({page_url: processed_text})
                            
                        time.sleep(1)
                
                    except Exception as e:
                        print(f"⚠️ Error fetching {page_url}: {e}")
            
            except:
                continue

# Save contents to a JSON file
with open("nasa_results_urls_content.json", "w", encoding="utf-8") as file:
    json.dump(urls_contents, file, indent=4)
