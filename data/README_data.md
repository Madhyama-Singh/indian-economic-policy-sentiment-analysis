# Data

The dataset used in this project is publicly available on Kaggle.

**Dataset Name:** India Financial News Sentiment Analysis
**Link:** https://www.kaggle.com/datasets/harshrkh/india-financial-news-headlines-sentiments/data
**Author:** harshrkh

## Dataset Details
- Raw shape: 2,00,500 rows × 5 columns
- Columns: Date, Title, URL, Sentiment, Confidence
- Date Range: January 1, 2017 — April 15, 2021
- News Sources: Economic Times, MoneyControl, Livemint,
  Business Today, Financial Express, NY Times, WSJ, Washington Post

## How to Use
1. Download the dataset from the Kaggle link above
2. Place the CSV file in this /data/ folder
3. Rename it to: News_sentiment_Jan2017_to_Apr2021.csv
4. Run the main analysis notebook

## Generated Files
After running the notebook, two CSV files will be generated
that are required by the Streamlit app:
- policy_sentiment_complete.csv
- daily_sentiment_index.csv

Place both in the root folder alongside app.py before
running the Streamlit application.

## Data Collection
Headlines were collected using the GDELT Headline Scrape
script by Prof. Ken Blake (https://drkblake.com/gdeltheadlinescrape/)
Sentiment labels were pre-generated using the Flair NLP tool.