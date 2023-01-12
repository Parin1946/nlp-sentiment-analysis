
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Download necessary NLTK data (only once)
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except nltk.downloader.DownloadError:
    nltk.download("wordnet")

class SentimentAnalyzer:
    """
    A class for performing sentiment analysis on text data.
    It includes methods for text preprocessing, feature extraction, and model training/evaluation.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.vectorizer = TfidfVectorizer(max_features=5000) # Limit features for efficiency
        self.model = MultinomialNB()

    def preprocess_text(self, text):
        """
        Cleans and preprocesses text by lowercasing, removing special characters,
        tokenizing, removing stopwords, and lemmatizing.
        """
        text = text.lower() # Lowercase
        text = re.sub(r"[^a-z\s]", "", text) # Remove special characters
        words = text.split() # Tokenize
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words] # Remove stopwords and lemmatize
        return " ".join(words)

    def train(self, X_train, y_train):
        """
        Trains the sentiment analysis model.

        Args:
            X_train (list): List of preprocessed text documents for training.
            y_train (list): List of corresponding sentiment labels.
        """
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vectorized, y_train)
        print("Model trained successfully.")

    def predict(self, text_list):
        """
        Predicts the sentiment of a list of text documents.

        Args:
            text_list (list): List of raw text documents to predict.

        Returns:
            list: Predicted sentiment labels.
        """
        preprocessed_texts = [self.preprocess_text(text) for text in text_list]
        X_vectorized = self.vectorizer.transform(preprocessed_texts)
        return self.model.predict(X_vectorized)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained model and prints a classification report.

        Args:
            X_test (list): List of preprocessed text documents for testing.
            y_test (list): List of true sentiment labels.
        """
        X_test_vectorized = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vectorized)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

if __name__ == "__main__":
    # Example Usage with dummy data
    print("Initializing Sentiment Analyzer...")
    analyzer = SentimentAnalyzer()

    # Dummy dataset (in a real scenario, load from CSV/DB)
    data = {
        "text": [
            "This movie was fantastic and I loved every moment of it!",
            "The service was terrible, very slow and rude staff.",
            "A decent effort, but it could have been much better.",
            "Absolutely brilliant, highly recommend!",
            "I hated the food, never coming back.",
            "It was okay, nothing special."
        ],
        "sentiment": ["positive", "negative", "neutral", "positive", "negative", "neutral"]
    }
    df = pd.DataFrame(data)

    # Preprocess texts
    df["processed_text"] = df["text"].apply(analyzer.preprocess_text)

    # Split data
    X = df["processed_text"].tolist()
    y = df["sentiment"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    print("Training model with dummy data...")
    analyzer.train(X_train, y_train)

    # Evaluate model
    print("Evaluating model...")
    analyzer.evaluate(X_test, y_test)

    # Make new predictions
    new_texts = [
        "This is an amazing product!",
        "I am so disappointed with this purchase.",
        "It's neither good nor bad."
    ]
    predictions = analyzer.predict(new_texts)
    print("\nNew Predictions:")
    for text, sentiment in zip(new_texts, predictions):
        print(f"Text: \"{text}\" -> Sentiment: {sentiment}")
