import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import string
import spacy
from tqdm import tqdm

# Ensure nltk resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize stopwords and SentimentIntensityAnalyzer
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

# Blacklist of words
blacklist = set(['https', 'com', 'nan', 'www', 'amp', 'png', 'http', 'would', 'like', 'iâ', '000', 'io'])

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text using spaCy's nlp pipeline
    tokens = nlp(text)
    
    # Remove stop words and non-alphabetic tokens, and perform lemmatization
    tokens = [token.lemma_ for token in tokens if token.is_alpha and token.text not in stop_words and token.text not in blacklist]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def apply_sentiment_analysis(text):
    if pd.isna(text):
        # Return a neutral sentiment score for NaN entries
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    return sia.polarity_scores(text)

def process_dataframe(df):
    # Preprocess title and selftext with progress bar
    tqdm.pandas(desc="Processing text")
    df['processed_title'] = df['title'].progress_apply(preprocess_text)
    df['processed_text'] = df['selftext'].progress_apply(preprocess_text)
    
    # Combine processed title and text
    df['combined_text'] = df['processed_title'] + ' ' + df['processed_text']
    
    # Apply sentiment analysis with progress bar
    tqdm.pandas(desc="Analyzing sentiment")
    df['sentiment'] = df['combined_text'].progress_apply(apply_sentiment_analysis)
    
    # Extract compound sentiment score
    df['sentiment_score'] = df['sentiment'].apply(lambda x: x['compound'])
    
    return df

def main(df):
    df = df.copy()
    # Ensure required columns exist
    required_columns = ['title', 'selftext', 'date_posted', 'ups']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")
    
    # Convert dates
    df['date_posted'] = pd.to_datetime(df['date_posted'], errors='coerce')
    
    # Process the dataframe
    print("Processing data...")
    processed_df = process_dataframe(df)
    
    # Calculate weighted sentiment score
    processed_df['weighted_sentiment'] = processed_df['sentiment_score'] * processed_df['ups']
    
    # Group by date and calculate weighted mean sentiment score
    result = processed_df.groupby(processed_df['date_posted'].dt.date).agg({
        'weighted_sentiment': 'sum',
        'ups': 'sum',
        'sentiment_score': 'mean'  # This is the unweighted mean, kept for comparison
    }).reset_index()
    
    # Calculate weighted mean sentiment score
    result['weighted_mean_sentiment'] = result['weighted_sentiment'] / result['ups']
    
    # Rename columns for clarity
    result = result.rename(columns={
        'date_posted': 'date',
        'sentiment_score': 'unweighted_mean_sentiment'
    })
    
    # Reorder columns
    result = result[['date', 'weighted_mean_sentiment', 'unweighted_mean_sentiment', 'ups']]
    
    # Get the latest date
    latest_date = result['date'].max().strftime('%Y-%m-%d')
    
    # Calculate overall weighted average sentiment score
    total_weighted_sentiment = (result['weighted_mean_sentiment'] * result['ups']).sum()
    total_ups = result['ups'].sum()
    overall_weighted_avg_sentiment = total_weighted_sentiment / total_ups
    
    # Create a filename with the latest date
    filename = f"reddit_pull/reddit_sentiment_{latest_date}.csv"
    
    # Save the result to a CSV file
    result.to_csv(filename, index=False)
    
    print(f"Data saved to {filename}")
    print(f"Overall weighted average sentiment score: {overall_weighted_avg_sentiment:.4f}")
    
    return result