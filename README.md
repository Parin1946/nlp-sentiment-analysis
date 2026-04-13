# NLP Sentiment Analysis

A Natural Language Processing project for sentiment analysis using Python, NLTK, and scikit-learn. Explores various text preprocessing techniques and machine learning models.

## Project Overview

This project focuses on building a sentiment analysis model to classify text as positive, negative, or neutral. It utilizes Python with popular NLP libraries such as NLTK for text preprocessing (tokenization, stemming, lemmatization) and scikit-learn for implementing various machine learning algorithms like Naive Bayes, SVM, and Logistic Regression.

## Features

-   **Text Preprocessing**: Tokenization, stop-word removal, stemming, and lemmatization.
-   **Feature Extraction**: TF-IDF, Bag-of-Words.
-   **Machine Learning Models**: Naive Bayes, SVM, Logistic Regression for sentiment classification.
-   **Model Evaluation**: Performance metrics (accuracy, precision, recall, F1-score) and confusion matrices.

## Getting Started

### Prerequisites

-   Python 3.8+
-   NLTK
-   scikit-learn
-   Pandas

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Parin1946/nlp-sentiment-analysis.git
    cd nlp-sentiment-analysis
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download NLTK data:
    ```python
    import nltk
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    ```

### Usage

1.  Prepare your dataset (e.g., CSV file with 'text' and 'sentiment' columns).
2.  Run the training script:
    ```bash
    python train_sentiment_model.py --data_path data/sentiment_data.csv
    ```
3.  Test the model with new text:
    ```bash
    python predict_sentiment.py --model_path models/sentiment_model.pkl --text "This movie was fantastic!"
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
