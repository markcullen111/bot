import os
from transformers import pipeline  # Uses Hugging Face's transformers library

class SentimentAnalyzer:
    """Analyzes sentiment of news headlines, tweets, and market discussions."""
    
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        Initializes the sentiment analysis model.
        Default model: Multilingual BERT sentiment analysis.
        """
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
            print("✅ Sentiment Analysis Model Loaded Successfully!")
        except Exception as e:
            print(f"⚠️ Error loading sentiment analysis model: {e}")
            self.sentiment_pipeline = None  # Model failed to load

    def analyze_sentiment(self, text):
        """
        Analyzes sentiment of the given text.
        Returns: {'label': sentiment_label, 'score': confidence}
        """
        if not self.sentiment_pipeline:
            return {"error": "Sentiment model not loaded."}

        try:
            result = self.sentiment_pipeline(text)
            return result[0]  # Returns label & score
        except Exception as e:
            print(f"⚠️ Error analyzing sentiment: {e}")
            return {"error": str(e)}

# Standalone test function
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Example test cases
    test_texts = [
        "The stock market is crashing!",
        "Bitcoin is reaching new all-time highs!",
        "This company has strong growth potential."
    ]

    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"Text: {text} -> Sentiment: {result}")

