#app.py
import numpy as np
from flask import Flask, render_template, request, jsonify
from rag import rag_pipeline
import json

app = Flask(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Set the custom encoder for Flask's jsonify
app.json_encoder = NumpyEncoder

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")
    try:
        # Get RAG response
        result = rag_pipeline(query)
        
        # Format human-readable response
        final_response = format_human_response(result, query)
        result["final_response"] = final_response
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Update in app.py
def format_human_response(result, query):
    """Format a human-readable response based on the RAG results"""
    response_parts = []
    
    # Handle different prediction types
    prediction_type = result.get("prediction_type")
    
    if prediction_type == "historical":
        date = result.get("date_queried")
        price_data = result.get("historical_price")
        
        if isinstance(price_data, dict) and "error" in price_data:
            response_parts.append(f"I couldn't retrieve the Bitcoin price for {date}. {price_data['error']}")
        else:
            if isinstance(price_data, dict):
                # Format full OHLCV data
                response_parts.append(f"Bitcoin data for {date}:")
                response_parts.append(f"Price: ${price_data['price']:,.2f} USD")
                if price_data.get('24h_high'):
                    response_parts.append(f"24h High: ${price_data['24h_high']:,.2f} USD")
                if price_data.get('24h_low'):
                    response_parts.append(f"24h Low: ${price_data['24h_low']:,.2f} USD")
                if price_data.get('volume'):
                    response_parts.append(f"Volume: ${price_data['volume']:,.2f} USD")
                if price_data.get('market_cap'):
                    response_parts.append(f"Market Cap: ${price_data['market_cap']:,.2f} USD")
            else:
                # Convert price to standard float if it's numpy float
                price = float(price_data) if hasattr(price_data, "item") else price_data
                response_parts.append(f"On {date}, Bitcoin was worth ${price:,.2f} USD.")
    
    elif prediction_type == "historical_range":
        days = result.get("days_queried")
        range_data = result.get("historical_range")
        
        if isinstance(range_data, dict) and "error" in range_data:
            response_parts.append(f"I couldn't retrieve Bitcoin prices for the last {days} days. {range_data['error']}")
        else:
            response_parts.append(f"Bitcoin prices for the last {days} days:")
            
            if isinstance(range_data, list) and len(range_data) > 0:
                # Create a table header
                response_parts.append("\nDate         | Open ($)   | High ($)   | Low ($)    | Close ($)  | Volume ($B)")
                response_parts.append("------------|------------|------------|------------|------------|-----------")
                
                # Add each day's data
                for day in range_data:
                    date = day['date']
                    open_price = f"${day['open']:,.2f}"
                    high = f"${day['high']:,.2f}"
                    low = f"${day['low']:,.2f}"
                    close = f"${day['close']:,.2f}"
                    volume = f"${day['volume']/1_000_000_000:.2f}B"
                    
                    response_parts.append(f"{date} | {open_price.ljust(10)} | {high.ljust(10)} | {low.ljust(10)} | {close.ljust(10)} | {volume}")
                
                # Add summary
                first_day = range_data[0]
                last_day = range_data[-1]
                price_change = ((last_day['close'] - first_day['open']) / first_day['open']) * 100
                
                response_parts.append(f"\nSummary: Bitcoin price changed by {price_change:.2f}% over this period.")
                response_parts.append(f"From ${first_day['open']:,.2f} on {first_day['date']} to ${last_day['close']:,.2f} on {last_day['date']}.")
            else:
                response_parts.append("No data available for this period.")
    
    elif prediction_type == "future":
        lstm_prediction = result.get("lstm_prediction", {})
        days_ahead = result.get("days_ahead", 1)
        
        # Get prediction info and convert to standard types if needed
        prediction = float(lstm_prediction.get("prediction")) if hasattr(lstm_prediction.get("prediction"), "item") else lstm_prediction.get("prediction")
        confidence = float(lstm_prediction.get("confidence")) if hasattr(lstm_prediction.get("confidence"), "item") else lstm_prediction.get("confidence")
        anomaly = lstm_prediction.get("anomaly_detection", {})
        metrics = lstm_prediction.get("model_metrics", {})
        
        # Get sentiment and volatility scores
        sentiment_score = float(lstm_prediction.get("sentiment_score")) if hasattr(lstm_prediction.get("sentiment_score"), "item") else lstm_prediction.get("sentiment_score")
        volatility_score = float(lstm_prediction.get("volatility_score")) if hasattr(lstm_prediction.get("volatility_score"), "item") else lstm_prediction.get("volatility_score")
        
        # Format time frame
        time_frame = "tomorrow"
        if days_ahead == 7:
            time_frame = "in a week"
        elif days_ahead > 1:
            time_frame = f"in {days_ahead} days"
        
        # Basic prediction
        response_parts.append(f"I predict Bitcoin will be worth ${prediction:,.2f} USD {time_frame}.")
        
        # Add confidence
        response_parts.append(f"Model confidence: {confidence}%")
        
        # Add accuracy info
        accuracy = float(metrics.get("accuracy_percentage")) if hasattr(metrics.get("accuracy_percentage"), "item") else metrics.get("accuracy_percentage")
        response_parts.append(f"Model accuracy: {accuracy}%")
        
        # Add sentiment and volatility info
        if sentiment_score is not None:
            sentiment_desc = "positive" if sentiment_score > 50 else "negative" if sentiment_score < 50 else "neutral"
            response_parts.append(f"Market sentiment: {sentiment_score}% ({sentiment_desc})")
        
        if volatility_score is not None:
            volatility_desc = "stable" if volatility_score > 75 else "moderate" if volatility_score > 50 else "volatile"
            response_parts.append(f"Market volatility: {volatility_score}% ({volatility_desc})")
        
        # Add market sentiment summary if available
        market_sentiment = result.get("market_sentiment_summary", {})
        if market_sentiment and isinstance(market_sentiment, dict):
            trend = market_sentiment.get("trend", "Unknown")
            certainty = market_sentiment.get("certainty", "Low")
            justification = market_sentiment.get("justification", "")
            response_parts.append(f"\n📊 Smart Contract Analysis: {trend} with {certainty} certainty")
            response_parts.append(f"📝 Justification: {justification}")
        
        # Add anomaly warning if present
        if anomaly.get("is_anomaly"):
            response_parts.append(f"⚠️ {anomaly.get('message')}")
        
        # Add range prediction
        high = float(lstm_prediction.get("24h_high")) if hasattr(lstm_prediction.get("24h_high"), "item") else lstm_prediction.get("24h_high")
        low = float(lstm_prediction.get("24h_low")) if hasattr(lstm_prediction.get("24h_low"), "item") else lstm_prediction.get("24h_low")
        response_parts.append(f"Expected range: ${low:,.2f} - ${high:,.2f} USD")
        
        # Add multi-day predictions if available
        multi_day = lstm_prediction.get("predictions_multi_day")
        if multi_day and days_ahead > 1:
            response_parts.append("\nDaily predictions:")
            for i, price in enumerate(multi_day):
                price = float(price) if hasattr(price, "item") else price
                response_parts.append(f"Day {i+1}: ${price:,.2f} USD")
    
    elif prediction_type == "current":
        current_price = result.get("current_price")
        
        if isinstance(current_price, dict) and "error" in current_price:
            response_parts.append(f"I couldn't retrieve the current Bitcoin price. {current_price['error']}")
        else:
            response_parts.append(f"The current Bitcoin price is ${current_price:,.2f} USD.")
    
    else:
        # If no prediction was made
        examples = result.get("query_examples", {})
        response_parts.append("I don't have an answer for that question about Bitcoin.")
        response_parts.append("\nHere are some example questions you can ask:")
        response_parts.append(f"- Historical price (single day): '{examples.get('historical_single')}'")
        response_parts.append(f"- Historical price (range): '{examples.get('historical_range')}'")
        response_parts.append(f"- Current price: '{examples.get('current')}'")
        response_parts.append(f"- Future price (short-term): '{examples.get('future_short')}'")
        response_parts.append(f"- Future price (medium-term): '{examples.get('future_medium')}'")
        response_parts.append(f"- Future price (long-term): '{examples.get('future_long')}'")
    
    return "\n".join(response_parts)

if __name__ == "__main__":
    app.run(debug=True)