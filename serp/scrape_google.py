import os
import sys
import sys
import pandas as pd
from tqdm import tqdm
import requests
import time
from multiprocessing import Pool
from newspaper import Article
import datetime
import pytz
import pickle
import pdb

current_date = datetime.datetime.now(
        pytz.timezone("America/New_York")
    ).strftime("%B %d, %Y")

sys.path.insert(0, '/home/al2644/research/codebase/wiki_entities_knowledge')
sys.path.insert(1, '/home/al2644/research/')

SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

def call_live_engine(query, search_type = 'web', num_results = 100):
    url = "https://api.serphouse.com/serp/live"
    params = {
        "api_token": SERPAPI_KEY,
        "q": query,
        "num_result": num_results,   
        "domain": "google.com",
        "lang": "en",  
        "device": "desktop",
        "serp_type": search_type,    # web or news
        "loc": "New York,United States"
    }
    response = requests.get(url, params=params)
    
    return response

def organic_search_engine (query, num_results = 20):
    response = call_live_engine(query, 'web', num_results)

    if response.status_code != 200:
        return {query: ''}
    
    try:
        organic_results = response.json()['results']['results']['organic']
        organic_results = [{'site_title': result['site_title'], 'snippet': result['snippet']} for result in organic_results]
        return {query: organic_results}
    
    except:
        return {query: ''}
