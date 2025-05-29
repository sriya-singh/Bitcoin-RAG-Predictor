#rag.py
import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import faiss
from sentence_transformers import SentenceTransformer
from keras.layers import InputLayer
from keras import Input
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model, save_model
import re
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
import yfinance as yf
import json
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()


# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "btc_lstm.keras")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "price_cache.json")

# Ensure model directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# LSTM Prep & Training
COINGECKO_API = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
CURRENCY = "usd"
DAYS = 365
PROJECTION_DAYS = 10



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Data Collection Functions
# These functions retrieve real-time data from various APIs and prepare it for analysis

# CoinGecko API implementation
def fetch_bitcoin_data():
    print("Fetching real-time data from CoinGecko API...")
    COINGECKO_API = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    CURRENCY = "usd"
    DAYS = 365  # 1 year of data

    url = f"{COINGECKO_API}?vs_currency={CURRENCY}&days={DAYS}&interval=daily"
    response = requests.get(url)
    data = response.json()

    # Convert JSON to DataFrame
    df = pd.DataFrame({
        "timestamp": [pd.to_datetime(p[0], unit='ms') for p in data['prices']],
        "open": [p[1] for p in data['prices']],
        "high": [p[1] for p in data['prices']],
        "low": [p[1] for p in data['prices']],
        "close": [p[1] for p in data['prices']],
        "market_cap": [m[1] for m in data['market_caps']],
        "volume": [v[1] for v in data['total_volumes']]
    })

    # Convert to structured JSON format
    json_data = []
    for i in range(len(df)):
        json_data.append({
            "timestamp": str(df["timestamp"][i]),
            "open": df["open"][i],
            "high": df["high"][i],
            "low": df["low"][i],
            "close": df["close"][i],
            "market_cap": df["market_cap"][i],
            "volume": df["volume"][i],
            "context": (
                f"On {df['timestamp'][i]}, Bitcoin opened at ${df['open'][i]:.2f}, "
                f"reached a high of ${df['high'][i]:.2f}, a low of ${df['low'][i]:.2f}, "
                f"and closed at ${df['close'][i]:.2f}. Market cap was ${df['market_cap'][i]:.2f}, "
                f"and trading volume was ${df['volume'][i]:.2f}."
            )
        })
    
    return json_data

# CryptoCompare API implementation
def fetch_cryptocompare_data():
    print("Fetching real-time data from CryptoCompare API...")
    # Load API key from environment
    API_KEY = os.getenv("CRYPTOCOMPARE_KEY")

    # Define parameters
    symbol = "BTC"
    currency = "USD"
    limit = 100  # Number of data points

    # API Endpoint
    url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol}&tsym={currency}&limit={limit}"

    # Headers with API key
    headers = {
        "authorization": f"Apikey {API_KEY}"
    }

    # Fetch data
    response = requests.get(url, headers=headers)
    cc_data = response.json()

    # Convert to DataFrame
    cc_df = pd.DataFrame(cc_data["Data"]["Data"])
    
    # Format data
    json_data = []
    for i, row in cc_df.iterrows():
        # Convert Unix timestamp to date
        dt = datetime.utcfromtimestamp(row['time']).strftime('%Y-%m-%d %H:%M:%S UTC')

        # Create snippet
        snippet = (
            f"On {dt}, Bitcoin opened at ${row['open']:.2f}, reached a high of ${row['high']:.2f}, "
            f"a low of ${row['low']:.2f}, and closed at ${row['close']:.2f}. "
            f"Trading volume (from) was {row['volumefrom']} and volume (to) was {row['volumeto']}. "
            f"Conversion type: {row.get('conversionType', 'N/A')}."
        )

        # Store structured data
        json_data.append({
            "timestamp": dt,
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volumefrom": row["volumefrom"],
            "volumeto": row["volumeto"],
            "conversionType": row.get("conversionType", "N/A"),
            "context": snippet,
            "time": row["time"]  # Keep original Unix timestamp for filtering
        })
    
    return json_data

# NewsAPI implementation
def fetch_news_data():
    print("Fetching real-time news data from NewsAPI...")
    NEWS_API_KEY = os.getenv("NEWS_KEY")

    # API URL
    URL = "https://newsapi.org/v2/everything"

    # Define search parameters
    params = {
        "q": "cryptocurrency price OR bitcoin price OR crypto",
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }

    # Fetch news data
    response = requests.get(URL, params=params)

    if response.status_code == 200:
        crypto_news = response.json()
        print("✅ Crypto news data fetched successfully!")
    else:
        print(f"❌ Error: {response.status_code}")
        return []

    # Extract articles
    articles = crypto_news.get("articles", [])

    # Filter for selected sources if needed
    selected_sources = {'Bitcoinist', 'Decrypt', 'Forbes', 'BBC News', 'CoinDesk', 'SecurityAffairs.com'}
    filtered_articles = [a for a in articles if a['source']['name'] in selected_sources]
    
    # Format news data
    json_data = []
    for article in filtered_articles:
        # Convert publishedAt timestamp to human-readable date
        try:
            published_date = datetime.fromisoformat(article['publishedAt'].replace("Z", "+00:00")).strftime('%Y-%m-%d')
            
            # Create snippet
            snippet = f"Source: {article['source']['name']}. On {published_date}, the article described: {article.get('description', 'No description available')}"
            
            # Store structured data
            json_data.append({
                "date": published_date,
                "source": article["source"]["name"],
                "title": article["title"],
                "description": article.get("description", ""),
                "url": article["url"],
                "context": snippet
            })
        except Exception as e:
            print(f"Error processing article: {e}")
    
    return json_data

# Helper for loading saved data (like Twitter data)
def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

# Orchestrates data collection from all sources
def gather_real_time_data():
    # Fetch all data in real-time
    bitcoin_market_data = fetch_bitcoin_data()
    cc_data = fetch_cryptocompare_data()
    # historical_data = fetch_historical_data()
    news_data = fetch_news_data()
    
    # Only twitter data is loaded from file as per your requirement
    bitcoin_tweets = load_json("data/twitter_sentiment_data.json")
    
    # Helper for temporal filtering
    def filter_recent_documents(documents_list, date_extractor=None):
        # Use current date as reference point
        current_date = datetime.now()
        two_years_ago = current_date - timedelta(days=730)  # 365*2 = 730 days
        filtered_docs = []
        
        for doc in documents_list:
            try:
                if date_extractor:
                    doc_date = date_extractor(doc)
                else:
                    # Try to find a date in the document using improved regex patterns
                    # First look for ISO format dates (YYYY-MM-DD)
                    date_match = re.search(r'\b(20\d{2})-(\d{1,2})-(\d{1,2})\b', str(doc))
                    if date_match:
                        year, month, day = map(int, date_match.groups())
                        if 1 <= month <= 12 and 1 <= day <= 31:  # Basic validation
                            doc_date = datetime(year, month, day)
                    else:
                        # Look for other date formats (MM/DD/YYYY or similar)
                        date_match = re.search(r'\b(\d{1,2})[/.-](\d{1,2})[/.-](20\d{2})\b', str(doc))
                        if date_match:
                            month, day, year = map(int, date_match.groups())
                            if 1 <= month <= 12 and 1 <= day <= 31:  # Basic validation
                                doc_date = datetime(year, month, day)
                        else:
                            # Look for years in isolation (2018, 2019, etc.)
                            year_match = re.search(r'\b(20\d{2})\b', str(doc))
                            if year_match:
                                year = int(year_match.group(1))
                                # If we only have a year, use January 1st of that year
                                doc_date = datetime(year, 1, 1)
                            else:
                                # If no date is found, include it as it might be recent
                                filtered_docs.append(doc)
                                continue
                
                print(f"Document date found: {doc_date}, threshold: {two_years_ago}")
                if doc_date >= two_years_ago:
                    filtered_docs.append(doc)
                else:
                    print(f"Excluded old document from {doc_date}")
            except Exception as e:
                print(f"Error parsing date: {e}")
                # If we can't parse the date reliably, we should be cautious
                # Only include if no clear old date is found
                if not re.search(r'\b20(1\d|2[0-3])\b', str(doc)):  # Exclude obvious old years
                    filtered_docs.append(doc)
        
        return filtered_docs

    # Custom date extractors
    def coingecko_date_extractor(entry):
        # Attempt to extract date from coingecko data
        date_match = re.search(r'\b(20\d{2})-(\d{1,2})-(\d{1,2})\b', str(entry))
        if date_match:
            return datetime.strptime(date_match.group(0), '%Y-%m-%d')
        return datetime.now()  # Default to current if not found

    def cc_data_extractor(entry):
        # Check if entry has a timestamp field
        if isinstance(entry, dict) and 'time' in entry:
            try:
                return datetime.fromtimestamp(entry['time'])
            except:
                pass
        return datetime.now()

    def tweet_date_extractor(entry):
        if isinstance(entry, dict) and 'created_at' in entry:
            try:
                return datetime.strptime(entry['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
            except:
                pass
        return datetime.now()

    # Apply filtering with custom date extractors where possible
    coingecko_data = filter_recent_documents(bitcoin_market_data, coingecko_date_extractor)
    cryptocompare_data = filter_recent_documents(cc_data, cc_data_extractor)
    # historical_data = filter_recent_documents(historical_data)
    news_data = filter_recent_documents(news_data)
    tweets_data = filter_recent_documents(bitcoin_tweets, tweet_date_extractor)

    # Print summary of filtered data
    print(f"Filtered data counts:")
    print(f"- CoinGecko: {len(coingecko_data)} (from {len(bitcoin_market_data)})")
    print(f"- CryptoCompare: {len(cryptocompare_data)} (from {len(cc_data)})")
    # print(f"- Historical: {len(historical_data)} (from {len(historical_data)})")
    print(f"- News: {len(news_data)} (from {len(news_data)})")
    print(f"- Tweets: {len(tweets_data)} (from {len(bitcoin_tweets)})")

    # Combine all documents
    documents = coingecko_data + cryptocompare_data + news_data + tweets_data
    
    return documents



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Vector Database and Retrieval Functions
# Functions that handle document embeddings and semantic search

# Semantic search with temporal awareness
def retrieve_context(query: str, top_k: int = 6):
    # Extract reference date from query - handle relative dates properly
    current_date = datetime.now()
    query_date = extract_date_from_query(query, current_date)
    date_str = query_date.strftime('%Y-%m-%d')
    print(f"Retrieved reference date from query: {date_str}")
    
    # If query has "last X days/weeks/months", calculate the date range
    days_back = 0
    if re.search(r'\b(last|past)\s+(\d+)\s+(day|days|week|weeks|month|months)\b', query, re.IGNORECASE):
        match = re.search(r'\b(last|past)\s+(\d+)\s+(day|days|week|weeks|month|months)\b', query, re.IGNORECASE)
        if match:
            number = int(match.group(2))
            unit = match.group(3).lower()
            
            # Convert to days
            if unit in ['day', 'days']:
                days_back = number
            elif unit in ['week', 'weeks']:
                days_back = number * 7
            elif unit in ['month', 'months']:
                days_back = number * 30
                
            # For "last X days" queries, adjust the start date
            start_date = current_date - timedelta(days=days_back)
            print(f"Query about last {days_back} days: from {start_date.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
    
    # Check if this is a future/projection query
    is_future_query = is_future_price_query(query)
    
    # Add date to query for better retrieval
    date_enhanced_query = f"{query} {date_str}"
    query_embedding = embedding_model.encode([date_enhanced_query])
    
    # First phase: semantic search
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k * 3)  # Get more initially
    indices = indices[0].tolist() if hasattr(indices[0], 'tolist') else indices[0]
    
    # Extract documents
    candidate_docs = [documents[i] for i in indices]
    
    # Second phase: temporal filtering
    temporally_valid_docs = []
    
    for doc in candidate_docs:
        doc_date = None
        # Extract date from document
        if isinstance(doc, dict):
            # Try various date fields
            for date_field in ['date', 'timestamp', 'created_at', 'time']:
                if date_field in doc:
                    try:
                        # Handle both string dates and timestamp numbers
                        if isinstance(doc[date_field], str):
                            # Try different formats
                            for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S UTC']:
                                try:
                                    doc_date = datetime.strptime(doc[date_field], fmt)
                                    break
                                except ValueError:
                                    continue
                        elif isinstance(doc[date_field], (int, float)):
                            # Assume Unix timestamp
                            doc_date = datetime.fromtimestamp(doc[date_field])
                        break
                    except:
                        pass
            
            # If no explicit date field, try to extract from context
            if not doc_date and 'context' in doc:
                date_match = re.search(r'On\s+(\d{4}-\d{2}-\d{2})', str(doc['context']))
                if date_match:
                    try:
                        doc_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                    except:
                        pass
        
        # If we found a date, check if it's valid for the query
        if doc_date:
            # For "last X days" queries, check if document is within the date range
            if days_back > 0:
                start_date = current_date - timedelta(days=days_back)
                if start_date <= doc_date <= current_date:
                    temporally_valid_docs.append(doc)
                    print(f"Included document from {doc_date} (within requested range)")
                else:
                    print(f"Excluded document from {doc_date} (outside requested range)")
            elif is_future_query:
                # For future queries, we want recent data, not older than 60 days from today
                sixty_days_ago = current_date - timedelta(days=60)
                if doc_date >= sixty_days_ago:
                    temporally_valid_docs.append(doc)
                else:
                    print(f"Excluded old document from {doc_date} (too old for future projection)")
            else:
                # For historical queries, documents must be before or on the query date
                # And not too old (not more than 180 days before query date)
                if doc_date <= query_date and doc_date >= (query_date - timedelta(days=180)):
                    temporally_valid_docs.append(doc)
                elif doc_date > query_date:
                    print(f"Excluded future document from {doc_date} (query date: {query_date})")
                else:
                    print(f"Excluded document from {doc_date} (too old relative to query date: {query_date})")
        else:
            # If no date found, we should be cautious
            # Include undated docs only if we need more context
            if len(temporally_valid_docs) < top_k / 2:
                temporally_valid_docs.append(doc)
    
    # Final step: extract context from filtered documents
    context_results = []
    for doc in temporally_valid_docs[:top_k]:  # Only take top_k documents
        # Extract the context field if it exists, otherwise use a string representation
        if isinstance(doc, dict) and 'context' in doc:
            context_results.append(doc['context'])
        else:
            # Try to extract key information from the document based on its structure
            if isinstance(doc, dict):
                # For dictionary documents, show key fields
                relevant_fields = ['date', 'timestamp', 'title', 'description', 'price', 'close']
                context_snippet = {}
                for field in relevant_fields:
                    if field in doc:
                        context_snippet[field] = doc[field]
                context_results.append(str(context_snippet))
            else:
                # For non-dictionary documents, use string representation
                context_results.append(str(doc))
    
    return context_results

# Get documents for sentiment analysis
def retrieve_sentiment_documents(query=None, reference_date=None):
    """
    Retrieves only CryptoCompare, News, and Tweets documents for sentiment analysis,
    excluding CoinGecko data. Filters by reference date if provided.
    
    Args:
        query (str, optional): The user's query to extract date information.
        reference_date (datetime, optional): Specific reference date to filter by.
    
    Returns:
        list: A list of temporally relevant documents from CryptoCompare, News, and Tweets sources.
    """
    # Get all documents
    all_documents = gather_real_time_data()
    
    # Extract date from query if reference_date not provided
    if not reference_date and query:
        reference_date = extract_date_from_query(query)
    
    # Default to current date if still None
    if not reference_date:
        reference_date = datetime.now()
        
    # Check if query has "last X days/weeks/months"
    days_back = 0
    if query:
        match = re.search(r'\b(last|past)\s+(\d+)\s+(day|days|week|weeks|month|months)\b', query, re.IGNORECASE)
        if match:
            number = int(match.group(2))
            unit = match.group(3).lower()
            
            # Convert to days
            if unit in ['day', 'days']:
                days_back = number
            elif unit in ['week', 'weeks']:
                days_back = number * 7
            elif unit in ['month', 'months']:
                days_back = number * 30
    
    # Filter out CoinGecko documents
    sentiment_docs = []
    
    for doc in all_documents:
        # Skip if not a dictionary
        if not isinstance(doc, dict):
            continue
            
        # Skip CoinGecko documents (have these fields but not volumefrom/volumeto)
        if ('open' in doc and 'high' in doc and 'low' in doc and 'close' in doc and 
            'market_cap' in doc and 'volume' in doc and 
            not ('volumefrom' in doc or 'volumeto' in doc)):
            continue
        
        # Extract date from document
        doc_date = None
        
        # Try various date fields
        for date_field in ['date', 'timestamp', 'created_at', 'time']:
            if date_field in doc:
                try:
                    # Handle both string dates and timestamp numbers
                    if isinstance(doc[date_field], str):
                        # Try different formats
                        for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S UTC']:
                            try:
                                doc_date = datetime.strptime(doc[date_field], fmt)
                                break
                            except ValueError:
                                continue
                    elif isinstance(doc[date_field], (int, float)):
                        # Assume Unix timestamp
                        doc_date = datetime.fromtimestamp(doc[date_field])
                    break
                except:
                    pass
        
        # If no explicit date field, try to extract from context
        if not doc_date and 'context' in doc:
            date_match = re.search(r'On\s+(\d{4}-\d{2}-\d{2})', str(doc['context']))
            if date_match:
                try:
                    doc_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                except:
                    pass
        
        # Apply temporal filtering
        if doc_date:
            # For "last X days" queries, check if document is within the date range
            if days_back > 0:
                start_date = reference_date - timedelta(days=days_back)
                if start_date <= doc_date <= reference_date:
                    sentiment_docs.append(doc)
                    # print(f"Included document from {doc_date} (within requested range for sentiment)")
            else:
                # For non-range queries, only include documents from before the reference date
                # And not too old (not more than 30 days before reference date)
                if doc_date <= reference_date and doc_date >= (reference_date - timedelta(days=30)):
                    sentiment_docs.append(doc)
                    # print(f"Included document from {doc_date} for sentiment (before reference date {reference_date})")
        else:
            # If no date found, include the document if it's a tweet or news
            if 'tweet_text' in doc or 'title' in doc or 'description' in doc:
                sentiment_docs.append(doc)
    
    # Print count of documents by type for debugging
    cryptocompare_count = len([d for d in sentiment_docs if 'volumefrom' in d and 'volumeto' in d])
    news_count = len([d for d in sentiment_docs if 'title' in d and 'description' in d])
    tweets_count = len([d for d in sentiment_docs if 'tweet_text' in d])
    
    print(f"Retrieved documents for sentiment analysis (reference date: {reference_date}):")
    print(f"- CryptoCompare: {cryptocompare_count}")
    print(f"- News: {news_count}")
    print(f"- Tweets: {tweets_count}")
    print(f"- Total: {len(sentiment_docs)}")
    
    return sentiment_docs



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Date and Time Analysis Functions
# Functions that extract and process temporal information from queries

# Parse dates from natural language
def extract_date_from_query(query, current_date=None):
    if current_date is None:
        current_date = datetime.now()
        
    # Try to extract date patterns like "YYYY-MM-DD", "MM/DD/YYYY", "Month DD, YYYY", etc.
    patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
        r'(\d{1,2}-\d{1,2}-\d{4})',  # DD-MM-YYYY
        r'(\w+ \d{1,2},? \d{4})'  # Month DD, YYYY
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            date_str = match.group(1)
            try:
                if "-" in date_str and len(date_str.split('-')[0]) == 4:
                    return datetime.strptime(date_str, '%Y-%m-%d')
                elif "-" in date_str:
                    return datetime.strptime(date_str, '%d-%m-%Y')
                elif "/" in date_str:
                    return datetime.strptime(date_str, '%m/%d/%Y')
                else:
                    # Try different month formats
                    for fmt in ['%B %d, %Y', '%B %d %Y', '%b %d, %Y', '%b %d %Y']:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except:
                            continue
            except:
                pass
    
    # Check for relative dates like "yesterday", "last week", etc.
    if re.search(r'\b(yesterday|last day)\b', query, re.IGNORECASE):
        return current_date - timedelta(days=1)
    elif re.search(r'\b(last week|a week ago)\b', query, re.IGNORECASE):
        return current_date - timedelta(days=7)
    elif re.search(r'\b(last month|a month ago)\b', query, re.IGNORECASE):
        return current_date - timedelta(days=30)
    
    # Additional time words check
    time_words = {
        'today': 0,
        'yesterday': 1,
        'last week': 7,
        'previous week': 7,
        'last month': 30,
        'recent': 14,  # Assume "recent" means last 2 weeks
        'current': 0,
        'this week': 3,
        'this month': 15,
    }
    
    # Check if any time words are in the query
    for time_word, days_ago in time_words.items():
        if re.search(rf'\b{time_word}\b', query, re.IGNORECASE):
            return current_date - timedelta(days=days_ago)
    
    # Check for "last X days/weeks/months" patterns
    range_match = re.search(r'\b(last|past)\s+(\d+)\s+(day|days|week|weeks|month|months)\b', query, re.IGNORECASE)
    if range_match:
        number = int(range_match.group(2))
        unit = range_match.group(3).lower()
        
        # Convert to days
        if unit in ['day', 'days']:
            return current_date - timedelta(days=number)
        elif unit in ['week', 'weeks']:
            return current_date - timedelta(days=number * 7)
        elif unit in ['month', 'months']:
            return current_date - timedelta(days=number * 30)
    
    # Default to current date if no time is specified
    return current_date

# Determine if query is about a past time range
def is_historical_range_query(query):
    patterns = [
        r'\b(what were|show me|display|historical)\b.*\b(bitcoin|btc)\b.*\b(prices|price)\b.*\b(last|past)\b.*\b(\d+)\b.*\b(days|day|weeks|week|months|month|years|year)\b',
        r'\b(bitcoin|btc)\b.*\b(prices|price)\b.*\b(last|past)\b.*\b(\d+)\b.*\b(days|day|weeks|week|months|month|years|year)\b'
    ]
    
    for pattern in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    
    return False

# Extract specific time periods from query
def extract_time_period(query):
    # Extract number of days/weeks/months
    period_patterns = [
        r'(\d+)\s+(day|days|week|weeks|month|months|year|years)',
        r'next\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)'
    ]
    
    for pattern in period_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            number = int(match.group(1))
            unit = match.group(2).lower()
            
            # Convert to days
            if unit in ['day', 'days']:
                return number
            elif unit in ['week', 'weeks']:
                return number * 7
            elif unit in ['month', 'months']:
                return number * 30
            elif unit in ['year', 'years']:
                return number * 365
    
    # Check for phrases like "next week", "next month"
    if re.search(r'\bnext\s+week\b', query, re.IGNORECASE):
        return 7
    elif re.search(r'\bnext\s+month\b', query, re.IGNORECASE):
        return 30
    elif re.search(r'\bnext\s+year\b', query, re.IGNORECASE):
        return 365
        
    return 1  # Default to 1 day if not specified

# Determine if query is about future prices
def is_future_price_query(query):
    future_patterns = [
        r'\b(will|going to|predict|forecast|future|tomorrow|next week|next month)\b',
        r'\bin\s+\d+\s+(day|days|week|weeks|month|months|year|years)\b'
    ]
    
    for pattern in future_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    
    return False

# Check if query is about price information
def is_price_query(query):
    price_patterns = [
        r'\b(price|cost|value|worth)\b',
        r'\bhow much\b.*\b(bitcoin|btc)\b',
        r'\b(bitcoin|btc)\b.*\b(price|cost|value|worth)\b'
    ]
    
    for pattern in price_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    
    return False

# Date comparison helper
def is_date_in_range(doc_date, query_date, window_days=3):
    """Check if a document date is within a window of days from the query date"""
    if not isinstance(doc_date, datetime):
        return False
    
    date_diff = abs((doc_date - query_date).days)
    return date_diff <= window_days



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Price Data Functions
# Functions to retrieve historical and current price information

# Get price for a specific past date
def get_historical_price(date):
    # Ensure date format is 'DD-MM-YYYY' for CoinGecko
    formatted_date = date.strftime('%d-%m-%Y')
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/history?date={formatted_date}&localization=false"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                "price": data.get('market_data', {}).get('current_price', {}).get('usd', None),
                "market_cap": data.get('market_data', {}).get('market_cap', {}).get('usd', None),
                "volume": data.get('market_data', {}).get('total_volume', {}).get('usd', None),
                "24h_high": data.get('market_data', {}).get('high_24h', {}).get('usd', None),
                "24h_low": data.get('market_data', {}).get('low_24h', {}).get('usd', None),
                "date": formatted_date
            }
        elif response.status_code == 429:
            return {"error": "API rate limit exceeded. Try again later."}
        else:
            return {"error": f"Failed to fetch data. Status code: {response.status_code}"}
    except Exception as e:
        return {"error": f"Error fetching data: {str(e)}"}

# Get price range for a period of days
def get_historical_price_range(days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    from_timestamp = int(start_date.timestamp())  # Ensure timestamps are in seconds
    to_timestamp = int(end_date.timestamp())
    
    # CoinGecko API URL for range data
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={from_timestamp}&to={to_timestamp}"

    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            # Format the data with OHLCV
            result = []
            # Group by day
            price_data = data['prices']
            volume_data = data['total_volumes']
            
            # Process daily data (simple approach - more sophisticated approach would use proper OHLC calculation)
            days_data = {}
            
            for price_point in price_data:
                timestamp = price_point[0]
                price = price_point[1]
                date_str = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d')
                
                if date_str not in days_data:
                    days_data[date_str] = {
                        'date': date_str,
                        'prices': [],
                    }
                
                days_data[date_str]['prices'].append(price)
            
            # Add volume data
            for vol_point in volume_data:
                timestamp = vol_point[0]
                volume = vol_point[1]
                date_str = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d')
                
                if date_str in days_data:
                    if 'volume' not in days_data[date_str]:
                        days_data[date_str]['volume'] = []
                    days_data[date_str]['volume'].append(volume)
            
            # Calculate OHLCV for each day
            result = []
            for date_str, day_data in sorted(days_data.items()):
                if 'prices' in day_data and day_data['prices']:
                    prices = day_data['prices']
                    result.append({
                        'date': date_str,
                        'open': prices[0],
                        'high': max(prices),
                        'low': min(prices),
                        'close': prices[-1],
                        'volume': max(day_data.get('volume', [0])) if 'volume' in day_data else 0
                    })
            
            return result
            
        elif response.status_code == 429:
            return {"error": "API rate limit exceeded. CoinGecko free tier limits have been reached. Please try again later."}
        else:
            return {"error": f"Failed to fetch historical data range. Status code: {response.status_code}"}
    except Exception as e:
        return {"error": f"Error fetching historical data range: {str(e)}"}

# Get the latest BTC price
def fetch_current_btc_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["bitcoin"]["usd"]
        elif response.status_code == 429:
            # Handle rate limit by returning the most recent price from our dataset
            print("API rate limit reached, using most recent recorded price")
            return float(df['close'].iloc[-1])
        else:
            print(f"Failed to fetch current price. Status code: {response.status_code}")
            # Return most recent price from dataset as fallback
            return float(df['close'].iloc[-1])
    except Exception as e:
        print(f"Error fetching current price: {str(e)}")
        # Return most recent price from dataset as fallback
        return float(df['close'].iloc[-1])



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LSTM Model Functions
# Functions for time-series prediction using LSTM neural networks

# Retrieve data for model training
def fetch_data(use_cached=False):  
    cache_file = os.path.join(DATA_DIR, "btc_price_data.csv")
    
    # Use cached data if available and requested
    if use_cached and os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=['timestamp'])
    
    # Otherwise fetch from API
    url = f"{COINGECKO_API}?vs_currency={CURRENCY}&days={DAYS}&interval=daily"
    try:
        data = requests.get(url).json()
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([p[0] for p in data['prices']], unit='ms'),
            "close": [p[1] for p in data['prices']]
        })
        
        # Cache the data
        df.to_csv(cache_file, index=False)
        return df
    except Exception as e:
        # If API fails, try to use cached data as fallback
        if os.path.exists(cache_file):
            print(f"API error: {str(e)}. Using cached data.")
            return pd.read_csv(cache_file, parse_dates=['timestamp'])
        else:
            raise Exception("Failed to fetch BTC price data and no cache available")

# Prepare data for LSTM model
def prepare_lstm_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close']].values)
    X, y = [], []
    for i in range(len(scaled) - PROJECTION_DAYS):
        X.append(scaled[i:i+PROJECTION_DAYS])
        y.append(scaled[i+PROJECTION_DAYS])
    return train_test_split(np.array(X), np.array(y), test_size=0.3, random_state=42), scaler

# Train the LSTM model
def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    return model

# Make price predictions
def predict_btc_price_lstm(days_ahead=1):
    last_prices = df['close'].values[-PROJECTION_DAYS:]
    scaled_input = scaler.transform(last_prices.reshape(-1, 1)).reshape(1, PROJECTION_DAYS, 1)
    
    # For multi-day prediction
    predictions = []
    current_input = scaled_input.copy()
    
    for i in range(days_ahead):
        prediction_scaled = lstm_model.predict(current_input)
        prediction_actual = scaler.inverse_transform(prediction_scaled)[0][0]
        predictions.append(prediction_actual)
        
        # Update input for next day prediction (remove oldest, add newest)
        current_input = np.append(current_input[:, 1:, :], prediction_scaled.reshape(1, 1, 1), axis=1)
    
    # Check for anomalies
    anomaly_info = detect_anomalies(predictions[0], last_prices)

    tweets_data = [t for t in documents if isinstance(t, dict) and 'tweet_text' in t]
    news_data = [n for n in documents if isinstance(n, dict) and 'title' in n and 'description' in n]
    
    context_data = tweets_data + news_data
    conf = calculate_final_confidence(df, y_test, y_pred, query=None)

    if anomaly_info["is_anomaly"]:
        conf = round(conf * anomaly_info["confidence_modifier"], 2)
    
    return {
        "prediction": round(predictions[0], 2),
        "predictions_multi_day": [round(p, 2) for p in predictions] if days_ahead > 1 else None,
        "24h_high": round(predictions[0] * 1.02, 2),
        "24h_low": round(predictions[0] * 0.98, 2),
        "confidence": conf,
        "model_metrics": model_metrics,
        "anomaly_detection": anomaly_info
    }



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Analysis and Metrics Functions
# Functions that calculate various metrics and perform analyses

# Calculate sentiment from documents
def calculate_sentiment_score(query=None, reference_date=None):
    """
    Calculate sentiment score based only on temporally relevant CryptoCompare, News, and Tweets data.
    
    Args:
        query (str, optional): The user's query to extract date information.
        reference_date (datetime, optional): Specific reference date to filter by.
        
    Returns:
        float: Sentiment score from 0-100, where 50 is neutral.
    """
    # Get temporally filtered documents for sentiment analysis (excluding CoinGecko)
    sentiment_docs = retrieve_sentiment_documents(query, reference_date)
    
    sentiments = []
    
    # Extract sentiment from tweets
    tweets_data = [t for t in sentiment_docs if 'tweet_text' in t]
    for tweet in tweets_data:
        sentiment = TextBlob(tweet['tweet_text']).sentiment.polarity
        sentiments.append(sentiment)
    
    # Extract sentiment from news
    news_data = [n for n in sentiment_docs if 'title' in n or 'description' in n]
    for news in news_data:
        # Use description or title or combine them
        news_text = news.get('description', '') or news.get('title', '')
        if news_text:  # Only analyze if there's text
            sentiment = TextBlob(news_text).sentiment.polarity
            sentiments.append(sentiment)
    
    # Extract sentiment from CryptoCompare data context fields
    cc_data = [c for c in sentiment_docs if 'context' in c and ('volumefrom' in c or 'volumeto' in c)]
    for item in cc_data:
        if 'context' in item:
            context_text = ""
            # Check if context is a list and convert to string if needed
            if isinstance(item['context'], list):
                context_text = ' '.join(str(element) for element in item['context'])
            else:
                context_text = str(item['context'])  # Ensure it's a string
                
            if context_text:  # Only process non-empty strings
                sentiment = TextBlob(context_text).sentiment.polarity
                sentiments.append(sentiment)
    
    # Calculate average sentiment score
    if sentiments:
        # Convert from [-1, 1] scale to [0, 100] scale where 50 is neutral
        score = np.mean(sentiments)
        normalized_score = round((score + 1) / 2 * 100, 2)
        
        print(f"Sentiment analysis summary:")
        print(f"- Total sources analyzed: {len(sentiments)}")
        print(f"- Raw sentiment score (scale -1 to 1): {score:.4f}")
        print(f"- Normalized sentiment score (0-100): {normalized_score}")
        
        return normalized_score
    else:
        print("Warning: No sentiment data available")
        return 50.0  # Return neutral sentiment if no data

# Calculate price volatility
def calculate_volatility(df):
    last_7_days = df['close'].values[-7:]
    volatility = np.std(last_7_days) / np.mean(last_7_days)
    return round((1 - volatility) * 100, 2)

# Calculate LSTM model performance
def calculate_model_metrics(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    accuracy = round(max(0, min(r2, 1)) * 100, 2)
    
    return {
        "r2_score": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "accuracy_percentage": accuracy
    }

# Calculate overall confidence
def calculate_final_confidence(df, y_test, y_pred, query=None, retrieved_context=None, 
                             sentiment_weight=0.3, volatility_weight=0.3, lstm_weight=0.4):
    """
    Calculate a final confidence score combining LSTM performance, market volatility, and sentiment analysis
    
    Args:
        df: DataFrame with price data
        y_test: Test target values
        y_pred: Predicted values
        query: User query for temporal reference
        retrieved_context: Retrieved context documents
        sentiment_weight: Weight for sentiment component
        volatility_weight: Weight for volatility component
        lstm_weight: Weight for LSTM accuracy component
    """
    # Extract reference date from query
    reference_date = None
    if query:
        reference_date = extract_date_from_query(query)
    
    # Calculate sentiment score using only non-CoinGecko data filtered by date
    sentiment_score = calculate_sentiment_score(query, reference_date)
    
    volatility_score = calculate_volatility(df)
    model_metrics = calculate_model_metrics(y_test, y_pred)
    lstm_confidence = model_metrics["accuracy_percentage"]
    
    # Normalize weights if they don't sum to 1
    total_weight = sentiment_weight + volatility_weight + lstm_weight
    if total_weight != 1.0:
        sentiment_weight /= total_weight
        volatility_weight /= total_weight
        lstm_weight /= total_weight
    
    final_confidence = (lstm_weight * lstm_confidence) + (sentiment_weight * sentiment_score) + (volatility_weight * volatility_score)
    
    # Additional confidence adjustment based on prediction anomaly
    recent_prices = df['close'].values[-5:]
    price_std = np.std(recent_prices)
    price_mean = np.mean(recent_prices)
    cv = price_std / price_mean  # Coefficient of variation
    
    # If market is highly volatile, reduce confidence slightly
    if cv > 0.05:  # 5% threshold for significant volatility
        final_confidence *= 0.95
    
    return round(final_confidence, 2)

# Detect unusual price movements
def detect_anomalies(prediction, history):
    last_price = history[-1]
    percent_change = ((prediction - last_price) / last_price) * 100
    
    # Define anomaly thresholds with confidence adjustment
    anomaly_thresholds = {
        "extreme": {
            "threshold": 15.0,
            "confidence_modifier": 0.7,
            "severity": "Extreme"
        },
        "high": {
            "threshold": 10.0,
            "confidence_modifier": 0.8,
            "severity": "High"
        },
        "moderate": {
            "threshold": 7.0,
            "confidence_modifier": 0.9,
            "severity": "Moderate"
        },
        "low": {
            "threshold": 5.0,
            "confidence_modifier": 0.95,
            "severity": "Low"
        }
    }
    
    # Check for anomalies by decreasing severity
    for severity, details in sorted(anomaly_thresholds.items(), 
                                   key=lambda x: x[1]["threshold"], 
                                   reverse=True):
        if abs(percent_change) > details["threshold"]:
            return {
                "is_anomaly": True,
                "percent_change": round(percent_change, 2),
                "direction": "increase" if percent_change > 0 else "decrease",
                "severity": details["severity"],
                "confidence_modifier": details["confidence_modifier"],
                "message": f"Anomaly detected: Predicted change of {round(percent_change, 2)}% is unusually large ({details['severity']} severity)"
            }
    
    # No anomaly detected
    return {
        "is_anomaly": False,
        "percent_change": round(percent_change, 2),
        "confidence_modifier": 1.0
    }

# Compare predictions with actual prices
def compare_predictions_with_live_data(prediction):
    current_price = fetch_current_btc_price()
    if isinstance(current_price, dict) and "error" in current_price:
        return current_price
    
    predicted = prediction["prediction"]
    diff = ((predicted - current_price) / current_price) * 100
    return {
        "current_market_price": current_price,
        "predicted_price": predicted,
        "difference": round(diff, 2),
        "trend_analysis": "Overestimated" if diff > 0 else "Underestimated"
    }



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LLM Integration Functions
# Functions that handle interaction with the LLM (TinyLlama in this case)

# Call the Ollama API
def generate_with_ollama(prompt: str):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "tinyllama",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print("✅ Ollama call successful")
        print("🧠 Model:", result.get("model"))
        print("📤 Response:", result.get("response"))
        return result.get("response", "[No response returned]")
    except requests.RequestException as e:
        print("❌ Ollama API call failed:", str(e))
        raise
    except json.JSONDecodeError:
        print("❌ Failed to decode JSON response from Ollama.")
        raise

# Generate explanations with LLM
def generate_ai_explanation(query, retrieved_context):
    # Extract date from query
    query_date = extract_date_from_query(query)
    date_str = query_date.strftime('%Y-%m-%d')
    
    # Check if we're looking at historical range query
    is_range = is_historical_range_query(query)
    
    # Create prompt based on type of query
    if is_range:
        days = extract_time_period(query)
        period_desc = f"the last {days} days" if days else "the recent period"
        
        ai_prompt = f"""You are an AI assistant specializing in Bitcoin price analysis. 
        Based on the following query and retrieved context only, provide a brief, insightful analysis:
        
        User Query: {query}
        
        Time Period Referenced: {period_desc}
        Query Date: {date_str}
        
        Retrieved Context (data up to the query date only): 
        {json.dumps(retrieved_context, indent=2, default=str)}
        
        Provide a concise explanation (3-5 sentences) that summarizes the Bitcoin price information
        for {period_desc}, based solely on the retrieved context above.
        Focus only on data from or before {date_str}. Never mention or refer to data after this date.
        Include specific price points and dates from the context. Only mention dates that appear in the context.
        """
    else:
        ai_prompt = f"""You are an AI assistant specializing in Bitcoin price analysis. 
        Based on the following query and retrieved context only, provide a brief, insightful analysis:
        
        User Query: {query}
        
        Time Period Referenced: {date_str} (extracted from query)
        
        Retrieved Context (data up to the query date only): 
        {json.dumps(retrieved_context, indent=2, default=str)}
        
        Provide a concise explanation (3-5 sentences) that helps the user understand the Bitcoin price information
        for the specific time period they mentioned, based solely on the retrieved context above.
        Focus only on data from or before {date_str}. Never mention or refer to data after this date.
        Address the specific time period the user is asking about. Do not reference any other data sources.
        """
    
    try:
        ai_explanation = generate_with_ollama(ai_prompt)
        return ai_explanation
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


def generate_market_sentiment_summary(context_data, sentiment_score, confidence):
    prompt = f"""
You are a financial AI assistant providing market sentiment to a liquidation smart contract.
 
Context: {json.dumps(context_data, indent=2)}
 
Sentiment Score: {sentiment_score}
Prediction Confidence: {confidence}
 
Based on the context, sentiment score, and confidence level, determine:
1. Is the market sentiment Bullish, Bearish, or Neutral?
2. What is your level of certainty (Low, Medium, High)?
3. Give a 1-sentence justification.
 
Respond in this JSON format:
{{
    "trend": "...",
    "certainty": "...",
    "justification": "..."
}}
"""
    try:
        result = generate_with_ollama(prompt)
        
        # Try to parse the JSON response
        try:
            parsed_result = json.loads(result)
            
            # Validate required fields exist
            if all(key in parsed_result for key in ["trend", "certainty", "justification"]):
                # Normalize the trend to one of the expected values
                if "bull" in parsed_result["trend"].lower():
                    parsed_result["trend"] = "Bullish"
                elif "bear" in parsed_result["trend"].lower():
                    parsed_result["trend"] = "Bearish"
                else:
                    parsed_result["trend"] = "Neutral"
                
                # Normalize the certainty to one of the expected values
                if "low" in parsed_result["certainty"].lower():
                    parsed_result["certainty"] = "Low"
                elif "high" in parsed_result["certainty"].lower():
                    parsed_result["certainty"] = "High"
                else:
                    parsed_result["certainty"] = "Medium"
                
                return parsed_result
            else:
                raise ValueError("Missing required fields in response")
                
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract information from text
            lines = [line.strip() for line in result.split('\n') if line.strip()]
            
            trend = "Neutral"
            certainty = "Low"
            justification = "Insufficient data for reliable sentiment analysis."
            
            for line in lines:
                lower_line = line.lower()
                if "bullish" in lower_line:
                    trend = "Bullish"
                elif "bearish" in lower_line:
                    trend = "Bearish"
                
                if "high certainty" in lower_line or "high confidence" in lower_line:
                    certainty = "High"
                elif "medium certainty" in lower_line or "medium confidence" in lower_line:
                    certainty = "Medium"
                
                # Look for potentially useful justification
                if len(line) > 20 and any(term in lower_line for term in ["because", "due to", "based on", "indicate", "suggest"]):
                    justification = line
            
            return {
                "trend": trend,
                "certainty": certainty,
                "justification": justification
            }
            
    except Exception as e:
        print(f"Error in generate_market_sentiment_summary: {str(e)}")
        return {
            "trend": "Neutral", 
            "certainty": "Low", 
            "justification": "Error analyzing market sentiment; defaulting to neutral position."
        }

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main RAG Pipeline Function
# Orchestrates the entire retrieval-augmented generation system

# Helper for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(i) for i in obj)
    else:
        return obj

# Main entry point that processes user queries
def rag_pipeline(query):
    # Retrieve context with the updated function
    retrieved_context = retrieve_context(query)
    
    # Base response
    response = {
        "retrieved_context": retrieved_context,
        "query_examples": {
            "historical_single": "What was the Bitcoin price on January 15, 2023?",
            "historical_range": "What were Bitcoin prices in the last 7 days?",
            "current": "What is the current Bitcoin price?",
            "future_short": "What will Bitcoin price be tomorrow?",
            "future_medium": "Predict Bitcoin price next 14 days",
            "future_long": "Forecast Bitcoin price in 30 days"
        }
    }
    
    # Check for historical range query first (has higher precedence)
    if is_historical_range_query(query):
        # Extract time period
        days = extract_time_period(query)
        if days:
            historical_range = get_historical_price_range(days)
            
            response["prediction_type"] = "historical_range"
            response["days_queried"] = days
            response["historical_range"] = historical_range
        else:
            response["error"] = "Couldn't determine the time period for historical data"
    
    # Check if query is about price (single date or future)
    elif is_price_query(query):
        # Extract date if present
        query_date = extract_date_from_query(query)
        
        # Check if the query explicitly asks about future prediction
        explicit_future_query = is_future_price_query(query)
        
        # If it's an explicit future price query or no date specified (assume current/future)
        if explicit_future_query or query_date is None:
            # Determine how many days ahead to predict
            days_ahead = extract_time_period(query)
            
            # Handle specific day counts
            day_match = re.search(r'in\s+(\d+)\s+(day|days)', query, re.IGNORECASE)
            if day_match:
                try:
                    days_ahead = int(day_match.group(1))
                    # Cap predictions at 60 days to maintain reasonable accuracy
                    days_ahead = min(days_ahead, 60)
                except:
                    pass
            
            # Use LSTM model for future prediction
            lstm_prediction = predict_btc_price_lstm(days_ahead)
            
            # Calculate sentiment score
            sentiment_score = calculate_sentiment_score(query)
            
            # Calculate volatility score
            volatility_score = calculate_volatility(df)
            
            # Calculate confidence based on retrieved context
            conf = calculate_final_confidence(df, y_test, y_pred, query)
            if lstm_prediction["anomaly_detection"]["is_anomaly"]:
                conf = round(conf * lstm_prediction["anomaly_detection"]["confidence_modifier"], 2)
            
            # Update the confidence score in the prediction
            lstm_prediction["confidence"] = conf
            
            # Add sentiment and volatility scores to prediction
            lstm_prediction["sentiment_score"] = sentiment_score
            lstm_prediction["volatility_score"] = volatility_score
            
            # Generate market sentiment summary for smart contract
            market_sentiment = generate_market_sentiment_summary(
                context_data=retrieved_context,
                sentiment_score=sentiment_score,
                confidence=conf
            )
            
            price_comparison = compare_predictions_with_live_data(lstm_prediction)
            
            response["prediction_type"] = "future"
            response["days_ahead"] = days_ahead
            response["lstm_prediction"] = lstm_prediction
            response["price_comparison"] = price_comparison
            response["market_sentiment_summary"] = market_sentiment
        
        # If it's a past price query with a specific date
        elif query_date:
            historical_price = get_historical_price(query_date)
            
            response["prediction_type"] = "historical"
            response["date_queried"] = query_date.strftime("%Y-%m-%d")
            response["historical_price"] = historical_price
        
        # If it seems like a current price query
        else:
            current_price = fetch_current_btc_price()
            
            response["prediction_type"] = "current"
            response["current_price"] = current_price
    
    # Convert all NumPy types to native Python types for JSON serialization
    response = convert_numpy_types(response)
    
    # Generate AI explanation/analysis based ONLY on the retrieved context
    try:
        ai_explanation = generate_ai_explanation(query, retrieved_context)
        response["ai_explanation"] = ai_explanation
    except Exception as e:
        response["ai_explanation_error"] = str(e)
    
    return response


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Initialize Models and Data
# This section runs when the module is imported, preparing models for use

# Load or train LSTM model once at startup
df = fetch_data()
(X_train, X_test, y_train, y_test), scaler = prepare_lstm_data(df)

if os.path.exists(MODEL_PATH):
    lstm_model = load_model(MODEL_PATH)
else:
    lstm_model = train_lstm_model(X_train, y_train)
    save_model(lstm_model, MODEL_PATH)

y_pred = lstm_model.predict(X_test)
model_metrics = calculate_model_metrics(y_test, y_pred)

# Initialize embedding model and document database
documents = gather_real_time_data()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedding_model.encode(documents)
dimension = doc_embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)
index.add(np.array(doc_embeddings, dtype=np.float32))