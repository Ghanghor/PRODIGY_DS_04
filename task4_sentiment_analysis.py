# This code will work once your editor can find the nltk library.
# If you get an error that the library "cannot be resolved", it means
# your code editor (like VS Code) is not looking at the right Python installation.
# Use 'Ctrl + Shift + P' -> 'Python: Select Interpreter' in VS Code to fix this.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- One-time NLTK downloads ---
# This ensures that the necessary NLTK data is available.
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK 'stopwords'...")
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK 'wordnet'...")
    nltk.download('wordnet')
# --------------------------------

# --- Main Script ---
try:
    # 1. LOAD THE UPLOADED DATASET
    file_path = 'twitter_training.csv'
    col_names = ['ID', 'Entity', 'Sentiment', 'Content']
    df = pd.read_csv(file_path, names=col_names)
    
    print(f"--- Dataset '{file_path}' loaded successfully ---")

    # 2. DATA CLEANING AND PREPROCESSING
    print("\n--- Cleaning and Preprocessing Text Data ---")
    df.dropna(subset=['Content'], inplace=True)
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#','', text)
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)

    df['Cleaned_Content'] = df['Content'].apply(clean_text)
    print("Text data has been cleaned.")

    # 3. SENTIMENT ANALYSIS AND VISUALIZATION
    print("\n--- Analyzing Sentiment Distribution ---")
    
    # Plot 1: Overall Sentiment Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Sentiment', data=df, order=['Positive', 'Negative', 'Neutral', 'Irrelevant'], palette='viridis')
    plt.title('Overall Sentiment Distribution', fontsize=16)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    plt.savefig('overall_sentiment_distribution.png')
    print("Plot 1: Overall sentiment distribution saved as 'overall_sentiment_distribution.png'")

    # Plot 2: Sentiment Distribution for Top Entities
    top_entities = df['Entity'].value_counts().nlargest(10).index
    df_top_entities = df[df['Entity'].isin(top_entities)]

    plt.figure(figsize=(12, 8))
    sns.countplot(y='Entity', hue='Sentiment', data=df_top_entities, order=top_entities, palette='plasma')
    plt.title('Sentiment Distribution for Top 10 Entities', fontsize=16)
    plt.xlabel('Number of Posts', fontsize=12)
    plt.ylabel('Entity', fontsize=12)
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig('top_entities_sentiment.png')
    print("Plot 2: Sentiment distribution for top entities saved as 'top_entities_sentiment.png'")

except FileNotFoundError:
    print(f"Error: '{file_path}' not found.")
    print("Please make sure your data file and this Python script are in the exact same folder.")
except Exception as e:
    print(f"An error occurred: {e}")
    