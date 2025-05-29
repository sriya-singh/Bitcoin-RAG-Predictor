# Bitcoin Price Prediction & Analysis RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that combines deep learning, sentiment analysis, and real-time data retrieval to provide comprehensive Bitcoin price predictions and market analysis.

![UI Screenshot](https://github.com/user-attachments/assets/2ff7fd29-1963-482c-86a7-47483fb9c0b1)

## 🎯 Overview

This system uses a multi-faceted approach to Bitcoin price analysis by combining:
- **LSTM neural networks** for time-series prediction
- **Vector databases** for semantic document retrieval
- **Sentiment analysis** from multiple data sources
- **Temporal awareness** for date-specific queries
- **Local LLM integration** for natural language explanations

## 🏗️ System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   RAG Pipeline  │    │  LLM Integration│
│                 │    │                 │    │                 │
│ • CoinGecko API │───▶│ • Vector DB     │───▶│ • TinyLlama     │
│ • CryptoCompare │    │ • FAISS Index   │    │ • Ollama API    │
│ • News APIs     │    │ • Embeddings    │    │ • Explanations  │
│ • Twitter/X     │    │ • Retrieval     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ LSTM Prediction │    │ Sentiment Score │    │ Final Response  │
│                 │    │                 │    │                 │
│ • Bidirectional │    │ • Multi-source  │    │ • Predictions   │
│ • Multi-layer   │    │ • Temporal      │    │ • Explanations  │
│ • Anomaly Det.  │    │ • TextBlob      │    │ • Confidence    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🧠 Technical Approach

### 1. Temporal-Aware Document Retrieval

**Problem**: Traditional RAG systems don't handle time-sensitive financial queries effectively.

**Solution**: Implemented temporal filtering that:
- Extracts dates from natural language queries
- Filters documents based on temporal relevance
- Handles relative dates ("last 7 days", "yesterday")
- Prevents data leakage from future dates

```python
def retrieve_context(query: str, top_k: int = 6):
    # Extract reference date from query
    query_date = extract_date_from_query(query, current_date)
    
    # Semantic search + temporal filtering
    query_embedding = embedding_model.encode([date_enhanced_query])
    distances, indices = index.search(query_embedding, top_k * 3)
    
    # Filter documents by temporal validity
    for doc in candidate_docs:
        if doc_date <= query_date and doc_date >= (query_date - timedelta(days=180)):
            temporally_valid_docs.append(doc)
```

### 2. Multi-Source Sentiment Analysis

**Challenge**: Bitcoin prices are heavily influenced by market sentiment across multiple platforms.

**Approach**: Aggregate sentiment from:
- **Social Media**: Twitter/X posts with Bitcoin mentions
- **News Articles**: Financial news headlines and descriptions  
- **Market Data**: CryptoCompare contextual information

```python
def calculate_sentiment_score(query=None, reference_date=None):
    sentiment_docs = retrieve_sentiment_documents(query, reference_date)
    
    sentiments = []
    # Process tweets, news, and market context
    for source in [tweets_data, news_data, cc_data]:
        sentiment = TextBlob(text).sentiment.polarity
        sentiments.append(sentiment)
    
    # Convert [-1,1] to [0,100] scale where 50 is neutral
    normalized_score = (np.mean(sentiments) + 1) / 2 * 100
    return normalized_score
```

### 3. Advanced LSTM Architecture

**Design**: Bidirectional LSTM with dropout regularization for robust time-series prediction.

```python
def train_lstm_model(X_train, y_train):
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=True)), 
        Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    return model
```

**Features**:
- **Bidirectional processing**: Captures patterns from both past and future context
- **Multi-day predictions**: Iterative forecasting for extended periods
- **Anomaly detection**: Identifies unusual price movements and adjusts confidence

### 4. Confidence Scoring System

**Methodology**: Weighted combination of multiple reliability indicators:

```python
def calculate_final_confidence(df, y_test, y_pred, query=None):
    sentiment_score = calculate_sentiment_score(query, reference_date)
    volatility_score = calculate_volatility(df)  
    lstm_confidence = calculate_model_metrics(y_test, y_pred)["accuracy_percentage"]
    
    # Weighted combination (configurable weights)
    final_confidence = (
        lstm_weight * lstm_confidence + 
        sentiment_weight * sentiment_score + 
        volatility_weight * volatility_score
    )
    
    return final_confidence
```

**Components**:
- **LSTM Performance**: R² score and prediction accuracy
- **Market Volatility**: Price stability over recent periods
- **Sentiment Alignment**: Market mood consistency
- **Anomaly Adjustment**: Reduced confidence for unusual predictions

### 5. Query Classification & Routing

**Smart Query Processing**: Automatically detects query intent and routes to appropriate handlers:

```python
def rag_pipeline(query):
    if is_historical_range_query(query):
        # "Bitcoin prices last 7 days"
        days = extract_time_period(query)
        historical_range = get_historical_price_range(days)
        
    elif is_future_price_query(query):
        # "Bitcoin price tomorrow" or "Predict BTC next week"
        days_ahead = extract_time_period(query)
        lstm_prediction = predict_btc_price_lstm(days_ahead)
        
    elif query_date in past:
        # "Bitcoin price on January 15, 2023"
        historical_price = get_historical_price(query_date)
```

## 📊 Data Sources & Integration

### Primary APIs
- **CoinGecko**: Historical and current price data
- **CryptoCompare**: OHLCV data with market context
- **News APIs**: Financial news sentiment
- **Social Media**: Twitter/X sentiment data

### Data Processing Pipeline
1. **Collection**: Real-time API polling and caching
2. **Normalization**: Standardized date formats and schemas
3. **Embedding**: Sentence transformers for semantic similarity
4. **Indexing**: FAISS for efficient vector search
5. **Temporal Filtering**: Date-aware document retrieval

## 🎮 Query Examples and Output

### Historical Queries
- "What was Bitcoin price on April 2, 2025?" [Output](https://drive.google.com/file/d/1qysVVABMKlbl7hLrZtTZ1gUV2A-dqpTv/view?usp=sharing)
- "What were Bitcoin prices in last 10 days" [Output](https://drive.google.com/file/d/1ROiOqa-7EKPF2t991Hj8h2MVgGlHl_qj/view?usp=sharing)

### Future Predictions  
- "What will Bitcoin price be tomorrow?" [Output](https://drive.google.com/file/d/1mqTFp3lmHASsXIhqvo8vB2IwipnPMhsh/view?usp=sharing)
- "Predict Bitcoin price next 14 days" [Output](https://drive.google.com/file/d/1HejojXn36TXTYU6ZCqwTb5QayCyi3wGL/view?usp=sharing)
- "Forecast Bitcoin price in 1 month" [Output](https://drive.google.com/file/d/1BJt-0y4pxJuADDwt4pYMCI6529kRHxtY/view?usp=sharing)

### Current Market
- "What is the current Bitcoin price?" [Output](https://drive.google.com/file/d/1OkFCQO7IUNftvUsmXum919BoMLmaNV0v/view?usp=sharing)

## 🔧 Installation & Setup

### Prerequisites
```bash
# Python dependencies
pip install tensorflow sentence-transformers faiss-cpu textblob pandas numpy requests

# Ollama for local LLM
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull tinyllama
```

### Configuration
```python
# API Configuration
COINGECKO_API = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
CRYPTOCOMPARE_API = "https://min-api.cryptocompare.com/data/v2/histoday"

# Model Parameters  
PROJECTION_DAYS = 30
CURRENCY = "usd"
DAYS = 365

# File Paths
MODEL_PATH = "./models/lstm_model.keras"
DATA_DIR = "./data/"
```

## 📈 Performance Metrics

### LSTM Model Performance
- **MAE**: 0.0359
- **RMSE**: 0.0452
- **Accuracy**: 97.83%

### Confidence Components
- **LSTM Weight**: 40% (prediction model accuracy)
- **Sentiment Weight**: 30% (market mood influence)  
- **Volatility Weight**: 30% (market stability factor)

### Anomaly Detection
- **Extreme**: >15% price change (70% confidence modifier)
- **High**: >10% price change (80% confidence modifier)
- **Moderate**: >7% price change (90% confidence modifier)
- **Low**: >5% price change (95% confidence modifier)

## 🚀 Key Innovations

1. **Temporal RAG**: First financial RAG system with robust temporal awareness
2. **Multi-Modal Confidence**: Combines technical, sentiment, and volatility indicators
3. **Anomaly-Aware Predictions**: Automatically detects and adjusts for unusual market conditions
4. **Query Intelligence**: Natural language understanding for diverse financial queries
5. **Real-Time Integration**: Live data incorporation with local LLM explanations

## 🔮 Use Cases

### Individual Investors
- Historical price research
- Short-term trend analysis  
- Sentiment-driven insights

### Trading Applications
- Automated signal generation
- Risk assessment integration
- Market sentiment tracking

### Research & Analysis
- Backtesting strategies
- Market behavior studies
- Sentiment correlation research

### Smart Contracts & DeFi
- Liquidation risk assessment
- Automated trading triggers
- Dynamic fee adjustments

## 🛡️ Limitations & Considerations

- **API Rate Limits**: CoinGecko free tier restrictions
- **Prediction Horizon**: Accuracy decreases for longer-term forecasts
- **Market Volatility**: Crypto markets are inherently unpredictable
- **Sentiment Lag**: Social sentiment may lag price movements
- **Data Dependency**: Quality depends on source data availability

## 📚 Research & References

This system builds upon research in:
- **Time-Series Forecasting**: LSTM networks for financial prediction
- **Sentiment Analysis**: TextBlob and transformer-based approaches
- **Information Retrieval**: Vector databases and semantic search
- **Financial ML**: Multi-modal confidence scoring systems

## 🤝 Areas for Improvement

- Additional data sources integration
- Advanced sentiment models (BERT, FinBERT)
- Ensemble prediction methods
- Real-time model updating
- Enhanced anomaly detection algorithms

---
